"""
src/ai_advisor/advisor.py
==========================
HuggingFace-powered loan advisory report generator.

Given a structured applicant context (from LoanContextBuilder) and a
vector store (VectorStore or TFIDFStore), this module:

  1. Retrieves the most relevant policy document excerpts for this applicant.
  2. Builds a structured prompt in Mistral instruction format.
  3. Calls the HuggingFace Inference API (mistralai/Mistral-7B-Instruct-v0.2).
  4. Returns a polished, styled Markdown advisory report.

Report structure
----------------
  # Loan Advisory Report
  Decision summary, confidence, risk tier
  ## Key Decision Factors
  Factors that drove the model's outcome
  ## Recommended Action Plan  (if rejected)
  Concrete, timeline-based improvement steps
  ## Policy References
  Excerpts from retrieved strategy documents (with source citation)
  ## Risk Profile Summary
  Formatted table of the applicant's key metrics

Design notes
------------
- No em dashes in the generated output (instructed in the system prompt).
- Tone is polished, business-friendly, and non-judgemental.
- The prompt is kept concise to stay within the model's context window while
  still providing enough context for a high-quality response.
- Generation parameters are conservative (temperature=0.4) to reduce
  hallucination risk for a high-stakes financial advisory context.

Public API
----------
  LoanAdvisor(vector_store, hf_model, hf_token)
    .advise(context, n_docs=5) -> str  (Markdown)
    .advise_and_save(context, output_path, n_docs=5) -> Path
"""
from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger as log

from config.settings import HF_MODEL, HF_TOKEN


# ---------------------------------------------------------------------------
# Prompt construction helpers
# ---------------------------------------------------------------------------

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
) -> str:
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
        "Each phase must contain at least two specific, actionable steps tailored to this applicant's profile."
        if outcome == "rejected"
        else "Include a brief section explaining what factors most strongly supported this approval."
    )

    system_block = textwrap.dedent("""\
        You are LAPAS, the Loan Approval Prediction and Advisory System.
        You produce detailed, professional advisory reports for loan applicants.
        Your tone is polished, clear, and business-friendly.
        You never use em dashes. Use commas, colons, or parentheses instead.
        You cite specific policy references from the provided document excerpts.
        You do not guarantee outcomes or make absolute promises.
        All monetary values are in South African Rand (R).
        Format your entire response as well-structured Markdown.
    """).strip()

    user_block = textwrap.dedent(f"""\
        Generate a detailed Loan Advisory Report for the following applicant.

        Applicant Reference: {applicant_id}
        Report Date: {datetime.now().strftime('%d %B %Y')}

        APPLICANT PROFILE:
        {applicant_summary}

        TOP MODEL DECISION DRIVERS (feature importance):
        {importance_summary}

        RELEVANT POLICY EXCERPTS:
        {policy_context}

        INSTRUCTIONS:
        - Open with a clear Decision Summary section stating the outcome ({outcome.upper()}),
          confidence ({confidence}), and risk tier ({risk_tier}).
        - Explain the Key Decision Factors in clear, non-technical language.
          Reference at least two of the top model drivers listed above.
        - {action_instruction}
        - Include a Policy References section that quotes or paraphrases the most
          relevant excerpt and cites the source document by name.
        - Close with a Risk Profile Summary table listing the applicant's key
          metrics alongside a plain-language assessment of each.
        - Do not use em dashes anywhere in your response.
        - Keep the language professional, clear, and empathetic.
    """).strip()

    # Mistral instruction format: <s>[INST] {system + user} [/INST]
    return f"<s>[INST] {system_block}\n\n{user_block} [/INST]"


# ---------------------------------------------------------------------------
# HuggingFace client wrapper
# ---------------------------------------------------------------------------

def _call_huggingface(prompt: str, model: str, token: str) -> str:
    """
    Call the HuggingFace Inference API and return the generated text.

    Uses huggingface_hub.InferenceClient, which handles authentication,
    retries, and streaming automatically.
    """
    try:
        from huggingface_hub import InferenceClient
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required.  Install it with: pip install huggingface-hub"
        ) from exc

    client = InferenceClient(model=model, token=token)

    log.info("Calling HuggingFace Inference API: model=%s", model)

    response = client.text_generation(
        prompt,
        max_new_tokens=1_800,
        temperature=0.4,       # lower temperature reduces hallucination risk
        top_p=0.92,
        repetition_penalty=1.1,
        do_sample=True,
        return_full_text=False,  # return only the newly generated tokens
    )

    log.info("HuggingFace response received (%d chars)", len(response))
    return response.strip()


# ---------------------------------------------------------------------------
# Fallback report (when HuggingFace is unavailable)
# ---------------------------------------------------------------------------

def _build_fallback_report(ctx: Dict[str, Any], retrieved_docs: List[str], retrieved_metas: List[Dict]) -> str:
    """
    Generate a structured Markdown report without calling the LLM.

    Used when the HuggingFace API is unavailable (no token, network error, etc.)
    so the system remains functional for demonstration and testing.
    """
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

    importance = ctx.get("feature_importance", [])
    importance_lines = "\n".join(
        f"- **{item['readable_name']}** (importance: {item['importance']:.4f})"
        for item in importance[:8]
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
        for doc, meta in zip(retrieved_docs[:2], retrieved_metas[:2]):
            source = meta.get("source_file", meta.get("source", "Policy Document"))
            excerpt = textwrap.shorten(doc, width=400, placeholder=" [...]")
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

## Key Decision Factors

The following features had the greatest influence on the model's assessment:

{importance_lines}
{action_section}
{policy_refs}

## Risk Profile Summary

| Metric | Value | Assessment |
|--------|-------|-----------|
{table_rows}

---

*This report was generated by the LAPAS Advisory System. It is intended as a
guidance tool and does not constitute a formal credit decision or legal advice.
For a formal credit assessment, contact a registered credit provider.*
"""


# ---------------------------------------------------------------------------
# Main advisor class
# ---------------------------------------------------------------------------

class LoanAdvisor:
    """
    Generates loan advisory Markdown reports using a retrieval-augmented LLM.

    Parameters
    ----------
    vector_store : a VectorStore or TFIDFStore instance (must support .query())
    hf_model     : HuggingFace model ID (default: from config.settings.HF_MODEL)
    hf_token     : HuggingFace API token (default: from config.settings.HF_TOKEN)
    use_llm      : if False, use the fallback rule-based report (for testing)
    """

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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def advise(
        self,
        context: Dict[str, Any],
        n_docs: int = 5,
    ) -> str:
        """
        Generate a Markdown advisory report for the given applicant context.

        Retrieves the n_docs most relevant policy document excerpts, builds a
        structured prompt, calls the HuggingFace LLM, and returns the result.
        Falls back to a rule-based report if the LLM call fails.

        Parameters
        ----------
        context : dict from LoanContextBuilder.build()
        n_docs  : number of policy excerpts to retrieve and include in the prompt

        Returns
        -------
        Markdown-formatted advisory report string
        """
        retrieved_docs, retrieved_metas = self._retrieve(context["query_text"], n_docs)

        if self.use_llm and self.hf_token:
            try:
                prompt = _build_prompt(context, retrieved_docs, retrieved_metas)
                report = _call_huggingface(prompt, self.hf_model, self.hf_token)
                # Ensure the response is well-formed markdown with a header
                if not report.strip().startswith("#"):
                    report = "# Loan Advisory Report\n\n" + report
                return report
            except Exception as exc:
                log.error(
                    "HuggingFace API call failed (%s).  "
                    "Falling back to rule-based report.", exc,
                )

        log.info("LoanAdvisor: generating fallback (rule-based) report.")
        return _build_fallback_report(context, retrieved_docs, retrieved_metas)

    def advise_and_save(
        self,
        context: Dict[str, Any],
        output_path: Union[str, Path],
        n_docs: int = 5,
    ) -> Path:
        """
        Generate an advisory report and write it to a Markdown file.

        Parameters
        ----------
        context     : dict from LoanContextBuilder.build()
        output_path : where to write the .md file
        n_docs      : number of policy excerpts to retrieve

        Returns
        -------
        Path to the written file
        """
        report = self.advise(context, n_docs=n_docs)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")

        log.success("Advisory report saved to %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

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
                "Retrieved %d policy excerpts for query: %s ...",
                len(docs), query_text[:80],
            )
            return docs, metas
        except Exception as exc:
            log.warning("Vector store retrieval failed (%s); using empty context.", exc)
            return [], []
