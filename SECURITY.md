# Security policy

## Supported versions

This repository is currently a research snapshot. Only the latest commit on
the default branch is considered for security fixes; no stable release series
is supported yet.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting feature for this
repository. Do not open a public issue containing credentials, personal data,
cluster details, or an unpatched exploit. Include the affected file or version,
reproduction steps, impact, and any suggested mitigation.

If private vulnerability reporting is unavailable, contact the repository
owner through their GitHub profile without including sensitive details in the
initial message.

## Model-file safety

The package currently loads trusted model artifacts with `joblib`. Joblib and
other pickle-compatible formats may execute code while loading. Only use model
files distributed from this repository, and do not replace them with artifacts
from an untrusted source.

## Scope

Scientific accuracy questions, unsupported molecular classes, and ordinary
prediction errors should be reported as regular issues. They are not security
vulnerabilities unless they create a separate confidentiality, integrity, or
availability impact.
