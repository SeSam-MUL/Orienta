# Security policy

## Supported versions

Orienta is research software maintained by a small team. Fixes, if any, are
normally made only for the newest release. Please check that the problem still
occurs with the
[latest version](https://github.com/SeSam-MUL/Orienta/releases/latest) before
reporting it.

## Reporting a vulnerability

Please **do not** open a public issue for a security problem.

Report it privately through GitHub instead:
[**Report a vulnerability**](https://github.com/SeSam-MUL/Orienta/security/advisories/new)
(also under the *Security* tab of this repository). If that is not possible for
you, e-mail [sebastian.samberger@unileoben.ac.at](mailto:sebastian.samberger@unileoben.ac.at) with "Orienta security" in the subject.

Please include the version, your operating system, and the steps that show the
problem. We aim to acknowledge reports and, if a fix is needed, to keep you
informed while it is prepared, but we cannot promise response or fix times.
Please give us reasonable time to release a fix before you publish details.

This policy describes how reports are handled. It is not a guarantee that Orienta
is free of vulnerabilities: the software is provided without warranty, as set out
in the [Disclaimer](README.md#disclaimer) and the [license](LICENSE).

## What is in scope

Orienta runs its analysis backend as a small web server on your own computer,
by default bound to `127.0.0.1`. It is meant to answer only Orienta itself and
scripts you run on the same machine. Relevant reports include, for example:

- a website open in your browser, or another computer on the network, that can make
  the backend read, write, delete or run something;
- a crafted data or crystal-structure file that makes Orienta run code or write
  outside the folders it should;
- a problem in the installers or in the update mechanism that could lead to
  installing something other than an Orienta release.
