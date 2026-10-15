# Contributing to Orienta

Thank you for your interest in Orienta. It is research software, and the most
useful contribution is often the simplest one: telling us where it fails on
your data.

## Ways to help

- **Report a problem.** Use the [bug report form][bug], or the
  **⚠ Report a problem** button inside the app, which collects the logs for you.
- **Ask a question.** [Discussions → Q&A][qa] is the place for "how do I…" and
  "does this result look right?". Answers stay searchable for the next person.
  They are given voluntarily and do not replace your own validation of your
  results (see the [Disclaimer](README.md#disclaimer)).
- **Suggest a feature.** Use the [feature request form][feature], or start in
  [Discussions → Ideas][ideas] if you would like to talk it through first.
- **Improve the documentation.** The user guide in [`docs/user-guide/`](docs/user-guide/)
  is plain Markdown. Corrections and clearer explanations are welcome as pull requests.
- **Contribute code.** See below.

## Reporting a problem

[Reporting a problem](docs/user-guide/ReportingProblems.md) explains what the
in-app report contains and how to create it. Two things make a report easy to act on:

- what you did (which file, which page, which action), and
- what you expected instead, especially when a result looks *wrong* rather than broken.

Issues are public. The problem report is designed not to include your measurement
files, but it does contain file and folder names (often with your user name and
sample names), log text, and optionally a screenshot of the window. Check every
file in the zip before you post it. If you would rather not post it, send it by
e-mail (see [Contact](#contact)). If something personal was posted by mistake,
tell us: we will remove it from the issue and, for uploaded files, ask GitHub Support
to delete them.

## Working on the code

### Setting up

Follow [INSTALL.md](INSTALL.md) for an installation from source: a conda
environment with Python 3.11, and Node.js 20.19 or newer (on the 22 line, 22.12 or newer) for the user interface.
Then:

```bash
python start_app.py              # backend + browser version, http://127.0.0.1:8000
cd frontend && npm run dev       # user interface with live reload, http://localhost:5173
```

The backend does not reload itself: restart it after changing Python code.
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) describes how the Electron shell,
the React interface, the FastAPI backend and the scientific libraries fit together.

### Tests

```bash
pytest tests/                    # Python backend
cd frontend && npm test          # user interface (Vitest)
```

Some backend tests are skipped because they need measurement files or a crystal
database that are not distributed with the repository. That is expected.

### Guidelines

- **One change per pull request.** Small, focused pull requests are reviewed faster.
- **Results need evidence.** If a change can alter a scientific result
  (orientations, phase assignment, quantification, maps), include a before/after
  comparison on real data and say which data. A synthetic test alone cannot show
  that a result is right.
- **Fail loudly.** When input does not match an assumption, raise a clear error
  instead of silently falling back to a default.
- **Dependencies.** Every new third-party Python import goes into `requirements.txt`.
- **No machine-specific values.** No absolute paths, drive letters or user names;
  derive paths from the project folder or from configuration.
- **Interface text** goes through the translation files in
  [`frontend/src/locales/`](frontend/src/locales/) (English, German, Japanese,
  Chinese). A test checks that all four languages have the same keys; if you
  cannot translate a string, add the English text and say so in the pull request.
- **Documentation.** A change that users will notice belongs on the matching page
  in `docs/user-guide/`.

### How changes are merged

The maintainers integrate changes and publish the releases. An accepted pull
request may therefore reach `main` as a separate commit; we normally credit you
there, for example with a `Co-authored-by` line.

## License

Orienta is licensed under [GPL-3.0-or-later](LICENSE). By contributing, you agree
that your contribution is licensed under the same terms. If your contribution
contains code from elsewhere, name its source and license in the pull request:
it must be compatible with the GPL-3.0 and will be listed in [NOTICE.md](NOTICE.md).

## Code of conduct

Everyone taking part in this project is expected to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Contact

For anything that should not be public, such as a problem report you would rather
not post, a question about collaboration, or a code-of-conduct concern, write to
Sebastian Samberger, Montanuniversität Leoben: [sebastian.samberger@unileoben.ac.at](mailto:sebastian.samberger@unileoben.ac.at).
E-mails and their attachments are used only to answer you or to handle your report.

Security problems: please follow [SECURITY.md](SECURITY.md).

[bug]: https://github.com/SeSam-MUL/Orienta/issues/new?template=bug_report.yml
[feature]: https://github.com/SeSam-MUL/Orienta/issues/new?template=feature_request.yml
[qa]: https://github.com/SeSam-MUL/Orienta/discussions/categories/q-a
[ideas]: https://github.com/SeSam-MUL/Orienta/discussions/categories/ideas
