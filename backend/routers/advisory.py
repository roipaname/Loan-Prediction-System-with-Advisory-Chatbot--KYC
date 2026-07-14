from fastapi import APIRouter, HTTPException

from database import operations as ops
from database.schemas import RetrieverTypeEnum
from backend.deps import get_context_builder, get_retriever
from src.ai_advisor.advisor import LoanAdvisor

router = APIRouter(tags=["advisory"])


@router.get("/predictions/{code}")
def get_prediction(code: str):
    app = ops.get_applicant_by_code(code)
    if app is None:
        raise HTTPException(status_code=404, detail="Applicant not found")

    pred = ops.get_latest_prediction(app.id)
    if pred is None:
        raise HTTPException(status_code=404, detail="No prediction stored for this applicant yet")

    model = ops.get_model(pred.model_id)
    return {
        "display_code":          code,
        "predicted_outcome":     pred.predicted_outcome.value,
        "approval_probability":  float(pred.approval_probability),
        "risk_tier":             pred.risk_tier,
        "shap_values":           pred.shap_values,
        "top_shap_features":     pred.top_shap_features,
        "model_algorithm":       model.algorithm.value if model else None,
    }


@router.post("/predictions/{code}/advisory")
def generate_advisory(code: str, retriever: str = "tfidf"):
    if retriever not in ("tfidf", "vector"):
        raise HTTPException(status_code=400, detail="retriever must be 'tfidf' or 'vector'")

    app = ops.get_applicant_by_code(code)
    if app is None:
        raise HTTPException(status_code=404, detail="Applicant not found")

    ctx = get_context_builder().build(applicant_id=app.id)
    store = get_retriever(retriever)
    report = LoanAdvisor(store).advise(ctx)

    pred = ops.get_latest_prediction(app.id)
    if pred is not None:
        ops.upsert_rag({
            "prediction_id":  pred.id,
            "retriever_type": RetrieverTypeEnum(retriever),
            "query_text":     ctx["query_text"],
            "llm_response":   report,
        })

    return {"display_code": code, "report": report, "retriever": retriever}
