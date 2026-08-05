import json

from fastapi import APIRouter, HTTPException

from database import operations as ops
from backend.deps import get_context_builder
from backend.schemas.applicant import ApplicationCreate, ApplicationResult, MAX_LOAN_TO_INCOME_RATIO
from backend.services.scoring_service import score_new_application

router = APIRouter(tags=["applicants"])


@router.get("/applicants")
def list_applicants(limit: int = 5000):
    """
    Full applicant + engineered-features + latest-prediction dataset, flat.
    The frontend filters/sorts/paginates this client-side, same as it did on
    frontend/utils/mock_data.get_data().
    """
    df = ops.get_applicants_flat(limit=limit)
    return json.loads(df.to_json(orient="records", date_format="iso"))


@router.get("/applicants/{code}")
def get_applicant(code: str):
    app = ops.get_applicant_by_code(code)
    if app is None:
        raise HTTPException(status_code=404, detail="Applicant not found")

    ctx = get_context_builder().build(applicant_id=app.id)
    ctx["prediction"]["risk_tier"] = ctx["prediction"]["risk_tier"].replace(" Risk", "")
    ctx["display_code"] = code
    return ctx


@router.post("/applicants", response_model=ApplicationResult)
def create_applicant(payload: ApplicationCreate):
    loan_pct = payload.loan_percent_income
    if loan_pct is None:
        loan_pct = payload.loan_amnt / max(payload.person_income, 1)
    if loan_pct > MAX_LOAN_TO_INCOME_RATIO:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Loan amount is {loan_pct*100:.0f}% of annual income, which exceeds the "
                f"maximum supported loan-to-income ratio of {MAX_LOAN_TO_INCOME_RATIO*100:.0f}%. "
                f"The classifier was not trained on ratios this extreme and its predictions "
                f"there are unreliable. Reduce the loan amount or increase income before "
                f"resubmitting."
            ),
        )
    return score_new_application(payload.model_dump())
