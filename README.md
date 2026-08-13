# Student Attendance Management System (CS402.3)

A runnable Python prototype for the supplied static-layout signing sheets. It uses classical computer vision to deskew the photographed form, detect the student table, locate the right-most signature column, classify each student as PRESENT/ABSENT, save processing stages, persist results in SQLite, and visualize attendance history.

## 1. Setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Process a signing sheet

```bash
python sams.py input/1.jpeg data/info.xml --date 2019-07-31
```

Outputs are written under `outputs/<image-name>/`:
- original/resized image
- deskewed image
- grayscale
- binary image
- detected table overlay
- final attendance overlay
- one cropped signature ROI per student
- `attendance.csv`
- `metadata.json`

Attendance is also written to `data/attendance.db`.

## 3. Process all supplied sheets

Example:

```bash
python sams.py input/1.jpeg data/info.xml --date 2019-07-31
python sams.py input/2.jpeg data/info.xml --date 2019-08-26
python sams.py input/3.jpeg data/info.xml --date 2019-08-28
python sams.py input/4.jpeg data/info.xml --date 2019-07-05
python sams.py input/5.jpeg data/info.xml --date 2019-07-12
```

## 4. Visualize a student's attendance

```bash
python infovis.py 10000409
```

Charts are saved to `outputs/charts/`.

## 5. Optional signature investigation

Use one extracted candidate ROI and one or more known reference signature images:

```bash
python investigate.py outputs/3/signature_rois/row_1_10000409.png reference1.png reference2.png
```

This is an exploratory ORB-feature comparison and should be presented as an advanced experiment, not as identity-proofing.

## 6. Run tests

```bash
pytest -q
```

## Architecture

- `src/attendance/vision.py`: image processing and attendance classification
- `xml_reader.py`: student XML mapping
- `database.py`: SQLite persistence
- `visualization.py`: attendance charts
- `verification.py`: optional signature comparison
- `sams.py`: main prototype CLI
- `infovis.py`: visualization CLI
- `investigate.py`: signature-investigation CLI

## Tuning

The default presence threshold is `0.020` (foreground/dark-pixel ratio inside the signature cell). It can be changed without editing code:

```bash
python sams.py input/1.jpeg data/info.xml --threshold 0.025
```

The detector assumes the coursework's static signing-sheet layout, which is appropriate to the stated scenario. If the physical form layout changes substantially, recalibration or a more general table detector will be required.

## 7. One-command supplied-data demo

```bash
python run_demo.py
```

This resets the demo database, processes all five supplied photographs, creates a student visualization, and runs the 30-cell evaluation.

## Supplied-data evaluation

`evaluate.py` contains manually inspected ground truth for the five uploaded sheets. Run:

```bash
python evaluate.py
```

The checked ground truth is only for evaluation of the supplied images; it is not used by the detector itself.

> `data/info.xml` in this package was reconstructed from the readable student rows in the uploaded signing sheets so the prototype can be run immediately. Replace it with the official `info.xml` when supplied by the module staff.

## 8. Optional proxy-signing review (new, backwards compatible)

The normal attendance detector is unchanged. Proxy-signing review is **opt-in** and never changes a student's `PRESENT`, `ABSENT`, or `UNCERTAIN` attendance result. It only adds an advisory signature-consistency status for staff review.

Use trusted reference signatures arranged by student index:

```text
references/
├── 10000409/
│   ├── ref_01.png
│   ├── ref_02.png
│   └── ref_03.png
└── 10009301/
    ├── ref_01.png
    └── ref_02.png
```

Run:

```bash
python sams.py input/5.jpeg data/info.xml --proxy-check --reference-dir references
```

If `--reference-dir` is omitted, the program uses earlier `PRESENT` signature ROI crops already stored in the same SQLite database. It requires at least two references by default, so the first sessions normally report `INSUFFICIENT_REFERENCE` rather than making a weak decision.

Possible advisory results:

- `CONSISTENT` — signature is reasonably consistent with trusted references.
- `NEEDS_REVIEW` — similarity is uncertain; a staff member should inspect it.
- `SUSPICIOUS` — unusually low similarity; manual verification is recommended.
- `INSUFFICIENT_REFERENCE` — not enough trusted samples yet.

Results are stored separately in the `signature_verifications` table and written to `proxy_verification.csv`. This separation is intentional: natural handwriting varies, so signature analysis must not automatically mark a student absent or accuse them of fraud.

The original exploratory command remains unchanged:

```bash
python investigate.py candidate.png ref1.png ref2.png
```

To use the newer conservative multi-reference assessment explicitly:

```bash
python investigate.py candidate.png ref1.png ref2.png ref3.png --proxy-assessment
```
