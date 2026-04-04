from sqlalchemy import create_engine
from sqlalchemy.orm import session, sessionmaker
from contextlib import contextmanager
from loguru import logger

from config.settings import DB_URL,DB_POOL_SIZE,DB_POOL_TIMEOUT,DB_MAX_OVERFLOW,DB_ECHO


class Connection:
    def __init__(self):
        self.db_url=DB_URL
        self.engine=self.init_db()
        self.local_session=None

    def init_db(self):
        """Create a database connection and session."""

        try:
            self.engine=create_engine(DB_URL,pool_size=DB_POOL_SIZE,max_overflow=DB_MAX_OVERFLOW,pool_timeout=DB_POOL_TIMEOUT,pool_pre_ping=True,echo=DB_ECHO)
            self.local_session=sessionmaker(bind=self.engine,expire_on_commit=True)
            from database.schemas import Base
            Base.metadata.create_all(bind=self.engine)
            logger.success("Database Initialized")
            return self.engine
        except Exception as e:
            logger.error(f"Failed to Initialize DB:{e}")
            raise


    @contextmanager
    def get_db(self):
        session=self.local_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database Error : {e}")
            raise
        finally:
            session.close()

if __name__ == "__main__":
    from database.schemas import (
        LoanApplicant,
        EngineeredFeatures,
        MLModel,
        ModelPrediction,
        PredictionOutcomeEnum,
        ModelAlgorithmEnum
    )

    import random

    # ✅ Initialize connection
    conn = Connection()

    try:
        with conn.get_db() as db:

            # =========================================================
            # 1. Create Dummy Loan Applicant
            # =========================================================
            applicant = LoanApplicant(
                person_age=30,
                person_gender="male",
                person_education="Bachelor",
                person_income=50000,
                person_emp_exp=5,
                person_home_ownership="RENT",
                loan_amnt=10000,
                loan_intent="PERSONAL",
                loan_grade="B",
                loan_int_rate=12.5,
                loan_percent_income=0.2,
                cb_person_cred_hist_length=4,
                credit_score=680,
                previous_loan_defaults_on_file=False,
                loan_status=1,
                source_split="train"
            )

            db.add(applicant)
            db.flush()  # get ID before commit

            print(f"Created Applicant: {applicant.id}")

            # =========================================================
            # 2. Engineered Features
            # =========================================================
            features = EngineeredFeatures(
                applicant_id=applicant.id,
                debt_to_income_ratio=0.25,
                loan_to_income_ratio=0.2,
                affordability_ratio=0.7,
                monthly_income=4000,
                monthly_loan_burden=800,
                is_high_risk=False,
                composite_risk_score=0.3
            )

            db.add(features)

            # =========================================================
            # 3. ML Model
            # =========================================================
            model = MLModel(
                algorithm=ModelAlgorithmEnum.random_forest,
                is_from_scratch=False,
                cv_accuracy=0.89,
                cv_precision=0.87,
                cv_recall=0.85,
                cv_f1_weighted=0.86,
                cv_auc_roc=0.91,
                is_champion=True
            )

            db.add(model)
            db.flush()

            print(f"Created Model: {model.id}")

            # =========================================================
            # 4. Prediction
            # =========================================================
            prediction = ModelPrediction(
                applicant_id=applicant.id,
                model_id=model.id,
                predicted_outcome=PredictionOutcomeEnum.approved,
                approval_probability=0.82,
                risk_tier="Low",
                shap_values={"income": 0.2, "credit_score": 0.5},
                top_shap_features=["credit_score", "income"]
            )

            db.add(prediction)

            print("Dummy data inserted successfully ✅")

    except Exception as e:
        print(f"Test failed ❌: {e}")




