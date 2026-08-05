# No eager re-exports here on purpose — importing VectorStore at package-init
# pulls in torch as a side effect of importing any submodule, and loading
# torch before the xgboost classifier deserializes segfaults on macOS Intel.
# See backend/deps.py. Always import from the submodule directly.
