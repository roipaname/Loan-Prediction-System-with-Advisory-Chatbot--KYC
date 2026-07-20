# Intentionally no eager re-exports here.
#
# Importing VectorStore at package-init time pulls in torch/sentence-transformers
# as a side effect of importing ANY submodule of this package (Python always
# runs __init__.py first). On this macOS Intel machine, if torch loads into the
# process before the xgboost-backed LoanClassifier is deserialized via joblib,
# that deserialization segfaults (confirmed: safe if xgboost loads first, torch
# after; unsafe in the reverse order — see backend/deps.py). Nothing in this
# codebase imports from the package root (always `from src.ai_advisor.<module>
# import ...`), so these re-exports were pure risk with no benefit.
