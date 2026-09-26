# The pipeline's version -- the single source for it. Semantic versioning
# (MAJOR.MINOR.PATCH; 0.x while this is a proof of concept): bump MINOR for a
# change in what is measured or how, PATCH for a fix that leaves the method
# alone. Every change gets an entry in CHANGELOG.md and a matching git tag
# (vX.Y.Z). It is stamped on the metrics CSV, the coach report, the app and
# MLflow runs, so any result can be traced back to the code that made it.
__version__ = "0.3.0"
