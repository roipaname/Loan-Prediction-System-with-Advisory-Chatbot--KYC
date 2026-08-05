"""
Generates the loan advisory report: retrieves relevant policy excerpts for
the applicant, builds a prompt, calls the HuggingFace LLM (falls back to a
rule-based report if that fails), returns Markdown.

Report sections: Decision Summary, Key Decision Factors, Recommended Action
Plan (if rejected), Policy References, Risk Profile Summary.
"""
from __future__ import annotations

import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger as log

from config.settings import HF_MODEL, HF_TOKEN


def _format_currency(value: float) -> str:
    return f"R{value:,.2f}"


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _credit_tier_label(score: int) -> str:
    if score >= 800: return "Exceptional (800 to 850)"
    if score >= 740: return "Very Good (740 to 799)"
    if score >= 670: return "Good (670 to 739)"
    if score >= 580: return "Fair (580 to 669)"
    return "Poor (300 to 579)"


def _build_applicant_summary(ctx: Dict[str, Any]) -> str:
    app  = ctx["applicant"]
    eng  = ctx["engineered"]
    pred = ctx["prediction"]

    score_label = _credit_tier_label(app.get("credit_score", 0))
    lines = [
        f"- Age: {app.get('age', 'N/A')}",
        f"- Gender: {app.get('gender', 'N/A')}",
        f"- Education: {app.get('education', 'N/A')}",
        f"- Annual Income: {_format_currency(app.get('annual_income', 0))}",
        f"- Monthly Income: {_format_currency(eng.get('monthly_income', 0))}",
        f"- Employment Experience: {app.get('employment_experience_years', 0)} years",
        f"- Employment Stability: {eng.get('employment_stability', 'N/A')}",
        f"- Home Ownership: {app.get('home_ownership', 'N/A')}",
        f"- Loan Amount Requested: {_format_currency(app.get('loan_amount', 0))}",
        f"- Loan Purpose: {str(app.get('loan_intent', 'N/A')).replace('_', ' ').title()}",
        f"- Interest Rate: {app.get('interest_rate_pct', 0):.2f}%",
        f"- Loan as Percentage of Income: {_format_pct(app.get('loan_percent_of_income', 0))}",
        f"- Credit Score: {app.get('credit_score', 0)} ({score_label})",
        f"- Credit History Length: {app.get('credit_history_years', 0):.1f} years",
        f"- Previous Loan Default on Record: {'Yes' if app.get('previous_default_on_record') else 'No'}",
        "",
        "Derived Risk Metrics:",
        f"- Debt-to-Income Ratio: {eng.get('debt_to_income_ratio', 0):.4f}",
        f"- Affordability Ratio: {eng.get('affordability_ratio', 0):.4f}",
        f"- Monthly Loan Burden: {_format_currency(eng.get('monthly_loan_burden', 0))}",
        f"- Composite Risk Score: {eng.get('composite_risk_score', 0):.4f} (0 = low risk, 1 = high risk)",
        f"- High Loan Burden Flag: {'Active' if eng.get('high_loan_burden_flag') else 'Not Active'}",
        f"- Thin Credit File: {'Yes' if eng.get('thin_credit_file') else 'No'}",
        f"- Young and Inexperienced: {'Yes' if eng.get('young_inexperienced') else 'No'}",
        f"- Credit Risk Interaction Flag: {'Active' if eng.get('credit_risk_interaction') else 'Not Active'}",
        f"- Overall High-Risk Classification: {'Yes' if eng.get('is_high_risk') else 'No'}",
        "",
        f"Model Decision: {pred.get('outcome', '').upper()}",
        f"Approval Probability: {pred.get('confidence', 'N/A')}",
        f"Risk Tier: {pred.get('risk_tier', 'N/A')}",
    ]
    return "\n".join(lines)


def _build_importance_summary(importance_list: List[Dict[str, Any]]) -> str:
    if not importance_list:
        return "Feature importance not available for this model."
    lines = []
    for item in importance_list[:10]:
        bar_len = int(item["importance"] / max(importance_list[0]["importance"], 1e-9) * 20)
        bar = "#" * bar_len
        lines.append(f"  {item['readable_name']:<38} {bar} ({item['importance']:.4f})")
    return "\n".join(lines)


def _build_policy_context(retrieved_docs: List[str], retrieved_metas: List[Dict]) -> str:
    if not retrieved_docs:
        return "No relevant policy documents retrieved."
    parts = []
    for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas), start=1):
        source = meta.get("source_file", meta.get("source", "Policy Document"))
        excerpt = textwrap.shorten(doc, width=600, placeholder=" [...]")
        parts.append(f"[Policy Excerpt {i} | Source: {source}]\n{excerpt}")
    return "\n\n".join(parts)


def _build_prompt(
    ctx: Dict[str, Any],
    retrieved_docs: List[str],
    retrieved_metas: List[Dict],
) -> tuple[str, str]:
    """Build the (system, user) message pair for the chat-completion call."""
    outcome      = ctx["prediction"]["outcome"]
    confidence   = ctx["prediction"]["confidence"]
    risk_tier    = ctx["prediction"]["risk_tier"]
    applicant_id = ctx.get("ref_id", "Unknown")

    applicant_summary  = _build_applicant_summary(ctx)
    importance_summary = _build_importance_summary(ctx.get("feature_importance", []))
    policy_context     = _build_policy_context(retrieved_docs, retrieved_metas)

    action_instruction = (
        "Include a Recommended Action Plan section with three timeline phases "
        "(Immediate: 0 to 3 months; Short-term: 3 to 12 months; Long-term: 12 months and beyond). "
        "Each phase must contain at least three specific, actionable steps tailored to this "
        "applicant's profile, and a short paragraph (2 to 3 sentences) explaining why each phase matters."
        if outcome == "rejected"
        else "Include a detailed section explaining what factors most strongly supported this "
             "approval, and a short paragraph of guidance on maintaining this favourable profile."
    )

    system_block = textwrap.dedent("""\
        You are LAPAS, the Loan Approval Prediction and Advisory System.
        You produce detailed, professional, in-depth advisory reports for loan applicants.
        Your tone is polished, clear, and business-friendly.
        You never use em dashes. Use commas, colons, or parentheses instead.
        You cite specific policy references from the provided document excerpts.
        You do not guarantee outcomes or make absolute promises.
        All monetary values are in South African Rand (R).
        Format your entire response as well-structured Markdown.
        Write in full explanatory paragraphs, not just short bullet fragments;
        bullets and tables are for lists and summaries only, every section needs
        substantive prose that explains the reasoning, not just states facts.
    """).strip()

    user_block = textwrap.dedent(f"""\
        Generate a thorough, detailed Loan Advisory Report for the following applicant.
        Target length: approximately 900 to 1300 words. Be comprehensive and specific
        to this applicant's numbers, do not write generic filler.

        Applicant Reference: {applicant_id}
        Report Date: {datetime.now().strftime('%d %B %Y')}

        APPLICANT PROFILE:
        {applicant_summary}

        TOP MODEL DECISION DRIVERS (feature importance):
        {importance_summary}

        RELEVANT POLICY EXCERPTS:
        {policy_context}

        INSTRUCTIONS:
        - Open with a Decision Summary section (2 to 3 paragraphs) stating the outcome
          ({outcome.upper()}), confidence ({confidence}), and risk tier ({risk_tier}), and
          giving the applicant a clear, plain-language sense of what this decision means for them.
        - Write a Key Decision Factors section that discusses at least four of the top model
          drivers listed above, one paragraph each, explaining in non-technical language why
          that factor matters to a credit decision and how this applicant's specific value
          compares to a healthy benchmark.
        - {action_instruction}
        - Include a Policy References section that quotes or paraphrases at least two of the
          most relevant excerpts and cites the source document by name, with a sentence
          connecting each excerpt back to this applicant's situation.
        - Close with a Risk Profile Summary table listing the applicant's key metrics alongside
          a plain-language assessment of each, followed by a short closing paragraph.
        - Do not use em dashes anywhere in your response.
        - Keep the language professional, clear, and empathetic, and avoid repeating the same
          sentence structure across sections.
    """).strip()

    return system_block, user_block


def _call_huggingface(
    system_block: str,
    user_block: str,
    model: str,
    token: str,
    max_retries: int = 1,
    retry_delay: float = 2.0,
    request_timeout: float = 25.0,
) -> str:
    """
    Call the HuggingFace Inference API via chat_completion() — not
    text_generation(), which providers now reject for instruct models like
    this one ("not supported for task text-generation").

    The free-tier backend can take ~25-30s to report "model is busy" rather
    than failing fast, so request_timeout bounds each attempt instead of
    letting a busy provider stall the whole request.
    """
    try:
        from huggingface_hub import InferenceClient
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required.  Install it with: pip install huggingface-hub"
        ) from exc

    client = InferenceClient(model=model, token=token, timeout=request_timeout)

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            log.info("Calling HuggingFace Inference API (chat_completion): model={} attempt={}",
                      model, attempt + 1)
            response = client.chat_completion(
                messages=[
                    {"role": "system", "content": system_block},
                    {"role": "user", "content": user_block},
                ],
                max_tokens=2_000,
                temperature=0.4,       # lower temperature reduces hallucination risk
                top_p=0.92,
            )
            text = response.choices[0].message.content
            log.info("HuggingFace response received ({} chars)", len(text))
            return text.strip()
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                log.warning("HuggingFace call attempt {} failed ({}); retrying in {}s...",
                            attempt + 1, exc, retry_delay)
                time.sleep(retry_delay)

    raise last_exc


def _build_financial_narrative(ctx: Dict[str, Any]) -> str:
    """Three paragraphs (affordability, credit, employment/housing) so the
    fallback report reads as prose, not just a bullet list and a table."""
    app = ctx["applicant"]
    eng = ctx["engineered"]

    income        = app.get("annual_income", 0)
    monthly_income = eng.get("monthly_income", 0)
    loan_amt      = app.get("loan_amount", 0)
    pct_income    = eng.get("loan_to_income_ratio", 0)
    afford        = eng.get("affordability_ratio", 0)
    monthly_burden = eng.get("monthly_loan_burden", 0)
    intent        = str(app.get("loan_intent", "N/A")).replace("_", " ").title()

    score  = app.get("credit_score", 0)
    tier   = _credit_tier_label(score).split(" (")[0]
    hist   = app.get("credit_history_years", 0)
    default = app.get("previous_default_on_record")

    exp_years = app.get("employment_experience_years", 0)
    home      = str(app.get("home_ownership", "N/A")).title()
    stability = str(eng.get("employment_stability", "N/A")).title()

    burden_multiple = (monthly_burden / monthly_income) if monthly_income > 0 else float("inf")
    if afford >= 0.6:
        afford_sentence = "the applicant retains a comfortable buffer of disposable income after debt service."
    elif afford > 0.0:
        afford_sentence = "the margin between income and estimated obligations is comparatively tight."
    else:
        afford_sentence = (
            f"the estimated repayment would consume {burden_multiple:.1f}x the applicant's entire monthly "
            f"income, meaning the loan is not affordable at the requested amount under any reasonable "
            f"assessment; the affordability ratio is floored at 0.00 rather than going negative."
        )

    p1 = (
        f"This application requests {_format_currency(loan_amt)} for {intent.lower()} purposes against "
        f"a declared annual income of {_format_currency(income)} ({_format_currency(monthly_income)} per "
        f"month). The requested amount represents {pct_income*100:.1f}% of annual income, which LAPAS "
        f"underwriting policy treats as "
        f"{'a loan-to-income concern above the standard 30% threshold' if pct_income > 0.30 else 'within the generally acceptable range below the 30% threshold'}. "
        f"On a monthly basis, the estimated loan repayment burden of {_format_currency(monthly_burden)} "
        f"corresponds to an affordability ratio of {afford:.2f}, meaning {afford_sentence}"
    )

    p2 = (
        f"The applicant's credit score of {score} falls in the {tier} tier, informed by a credit history "
        f"spanning {hist:.1f} years. "
        f"{'A longer credit history gives lenders more data points to assess repayment behaviour over time, which supports this application.' if hist >= 4 else 'A relatively short credit history limits the amount of repayment behaviour data available, a factor the model weighs conservatively.'} "
        f"{'No previous loan default is recorded for this applicant, one of the strongest positive signals in the LAPAS risk model.' if not default else 'A previous loan default is recorded on this applicant, one of the most significant negative signals in the LAPAS risk model, and it materially increases assessed risk.'}"
    )

    p3 = (
        f"On the employment and housing side, the applicant reports {exp_years} years of employment "
        f"experience and a home ownership status of {home}, which the model classifies as "
        f"{stability.lower()} employment stability. "
        f"{'Stable, longer-tenured employment is generally associated with more predictable income and lower default risk, which is factored favourably into the overall assessment.' if stability.lower() == 'stable' else 'Shorter or less continuous employment tenure introduces additional uncertainty into the income stability assessment, which the model factors conservatively.'}"
    )

    return "\n\n".join([p1, p2, p3])


def _build_fallback_report(ctx: Dict[str, Any], retrieved_docs: List[str], retrieved_metas: List[Dict]) -> str:
    """Rule-based Markdown report, used when the LLM is unavailable."""
    pred    = ctx["prediction"]
    app     = ctx["applicant"]
    eng     = ctx["engineered"]
    outcome = pred["outcome"]
    ref_id  = ctx.get("ref_id", "N/A")

    outcome_line = (
        "**APPROVED**" if outcome == "approved"
        else "**REJECTED**"
    )
    decision_colour = "Approval granted" if outcome == "approved" else "Application not approved at this time"

    financial_narrative = _build_financial_narrative(ctx)

    importance = ctx.get("feature_importance", [])
    importance_lines = "\n".join(
        f"- **{item['readable_name']}** (importance: {item['importance']:.4f})"
        for item in importance[:12]
    )

    action_section = ""
    if outcome == "rejected":
        steps: List[str] = []
        if eng.get("is_high_risk") or eng.get("composite_risk_score", 0) > 0.5:
            steps.append("Address the overall high-risk classification by resolving the most critical risk flags identified below.")
        if app.get("previous_default_on_record"):
            steps.append("Resolve any outstanding loan defaults. Contact your lender to negotiate a settlement or structured repayment plan.")
        if eng.get("high_loan_burden_flag"):
            ratio = eng.get("loan_to_income_ratio", 0)
            steps.append(f"Reduce the requested loan amount. Your current loan-to-income ratio is {ratio:.2f}; bring it below 0.25 by requesting a smaller amount.")
        if eng.get("thin_credit_file"):
            steps.append("Build your credit history. Open a secured credit facility and manage it consistently for 12 to 24 months.")
        score = app.get("credit_score", 0)
        if score < 670:
            steps.append(f"Improve your credit score (currently {score}). Pay all obligations on time and reduce revolving credit utilisation below 30%.")
        if eng.get("employment_stability") == "unstable":
            steps.append("Maintain continuous employment with your current employer for at least 12 months to achieve the stable employment classification.")
        if not steps:
            steps.append("Review the risk factors identified above and address each one systematically before reapplying.")
        steps.append("Allow a minimum of 3 months before reapplying to allow meaningful improvement to be reflected in your profile.")

        action_section = "\n\n## Recommended Action Plan\n\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))

    policy_refs = ""
    if retrieved_docs:
        policy_refs = "\n\n## Policy References\n\n"
        for doc, meta in zip(retrieved_docs[:3], retrieved_metas[:3]):
            source = meta.get("source_file", meta.get("source", "Policy Document"))
            excerpt = textwrap.shorten(doc, width=550, placeholder=" [...]")
            policy_refs += f"> {excerpt}\n>\n> *Source: {source}*\n\n"

    risk_rows = [
        ("Credit Score", f"{app.get('credit_score', 0)}", _credit_tier_label(app.get("credit_score", 0))),
        ("Loan-to-Income Ratio", f"{eng.get('loan_to_income_ratio', 0):.4f}", "High concern" if eng.get("high_loan_burden_flag") else "Acceptable"),
        ("Affordability Ratio", f"{eng.get('affordability_ratio', 0):.4f}", "Marginal" if eng.get("affordability_ratio", 1) < 0.6 else "Adequate"),
        ("Composite Risk Score", f"{eng.get('composite_risk_score', 0):.4f}", "High" if eng.get("is_high_risk") else "Moderate" if eng.get("composite_risk_score", 0) > 0.4 else "Low"),
        ("Employment Stability", str(eng.get("employment_stability", "N/A")).title(), "Concern" if eng.get("employment_stability") == "unstable" else "Good"),
        ("Previous Default on Record", "Yes" if app.get("previous_default_on_record") else "No", "Critical concern" if app.get("previous_default_on_record") else "None"),
        ("Thin Credit File", "Yes" if eng.get("thin_credit_file") else "No", "Concern" if eng.get("thin_credit_file") else "None"),
    ]

    table_rows = "\n".join(f"| {r[0]} | {r[1]} | {r[2]} |" for r in risk_rows)

    return f"""# Loan Advisory Report

**Applicant Reference:** {ref_id}
**Report Date:** {datetime.now().strftime('%d %B %Y')}
**Model:** {ctx.get('model_algorithm', 'N/A').replace('_', ' ').title()}

---

## Decision Summary

{outcome_line} | Confidence: {pred.get('confidence', 'N/A')} | Risk Tier: {pred.get('risk_tier', 'N/A')}

{decision_colour}. Your application has been assessed by the LAPAS predictive model, which evaluated your financial profile across multiple dimensions. The sections below explain the key factors that influenced this decision.

---

## Financial Profile Analysis

{financial_narrative}

---

## Key Decision Factors

The following features had the greatest influence on the model's assessment, ranked by importance:

{importance_lines}
{action_section}
{policy_refs}

## Risk Profile Summary

| Metric | Value | Assessment |
|--------|-------|-----------|
{table_rows}

---

{decision_colour}, based on the factors summarised above. This assessment reflects the applicant's
profile at the time of submission; a materially different financial position, credit history update,
or reapplication with adjusted terms may produce a different outcome. This report was generated by the
LAPAS Advisory System and is intended as a guidance tool, it does not constitute a formal credit decision
or legal advice. For a formal credit assessment, contact a registered credit provider.
"""


class LoanAdvisor:
    """Generates loan advisory Markdown reports using a retrieval-augmented LLM.
    vector_store must support .query() (VectorStore or TFIDFStore).
    use_llm=False skips the LLM and always uses the rule-based report."""

    def __init__(
        self,
        vector_store: Any,
        hf_model: Optional[str] = None,
        hf_token: Optional[str] = None,
        use_llm: bool = True,
    ) -> None:
        self.vector_store = vector_store
        self.hf_model     = hf_model or HF_MODEL
        self.hf_token     = hf_token or HF_TOKEN
        self.use_llm      = use_llm

        if use_llm and not self.hf_token:
            log.warning(
                "LoanAdvisor: HF_API_TOKEN is not set.  "
                "LLM calls will fail.  Set HF_API_TOKEN in .env or pass hf_token= "
                "to LoanAdvisor(), or construct with use_llm=False for the fallback report."
            )

    def advise(
        self,
        context: Dict[str, Any],
        n_docs: int = 5,
    ) -> str:
        """Generate a Markdown advisory report for this applicant context
        (from LoanContextBuilder.build()). Falls back to a rule-based report
        if the LLM call fails."""
        retrieved_docs, retrieved_metas = self._retrieve(context["query_text"], n_docs)

        if self.use_llm and self.hf_token:
            try:
                system_block, user_block = _build_prompt(context, retrieved_docs, retrieved_metas)
                report = _call_huggingface(system_block, user_block, self.hf_model, self.hf_token)
                # Ensure the response is well-formed markdown with a header
                if not report.strip().startswith("#"):
                    report = "# Loan Advisory Report\n\n" + report
                return report
            except Exception as exc:
                log.error(
                    "HuggingFace API call failed ({}).  Falling back to rule-based report.",
                    exc,
                )

        log.info("LoanAdvisor: generating fallback (rule-based) report.")
        return _build_fallback_report(context, retrieved_docs, retrieved_metas)

    def advise_and_save(
        self,
        context: Dict[str, Any],
        output_path: Union[str, Path],
        n_docs: int = 5,
    ) -> Path:
        """Generate an advisory report and write it to output_path as Markdown."""
        report = self.advise(context, n_docs=n_docs)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")

        log.success("Advisory report saved to %s", output_path)
        return output_path

    def _retrieve(
        self,
        query_text: str,
        n_docs: int,
    ) -> tuple[List[str], List[Dict]]:
        """Query the vector store and return flat lists of docs and metadata."""
        try:
            result = self.vector_store.query([query_text], n_results=n_docs)
            docs  = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[{}] * len(docs)])[0]
            log.info(
                "Retrieved {} policy excerpts for query: {} ...",
                len(docs), query_text[:80],
            )
            return docs, metas
        except Exception as exc:
            log.warning("Vector store retrieval failed ({}); using empty context.", exc)
            return [], []
