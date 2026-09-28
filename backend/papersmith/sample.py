"""A demo ledger so the app can be tried end to end. All data here is fictional."""

from __future__ import annotations

from .models import Claim, ClaimType as T, DataTable, Figure, Ledger, Reference

DEMO_BIB = """@article{example2021survey,
  title   = {Example survey of road surface monitoring (fictional reference)},
  author  = {Author, A. and Writer, B.},
  journal = {Journal of Examples},
  year    = {2021}
}

@inproceedings{example2022vision,
  title     = {Example camera-based pothole detector (fictional reference)},
  author    = {Researcher, C.},
  booktitle = {Proceedings of the Example Conference},
  year      = {2022}
}

@article{example2020cost,
  title   = {Example study of road maintenance costs (fictional reference)},
  author  = {Analyst, D.},
  journal = {Example Transport Review},
  year    = {2020}
}"""


def demo_ledger() -> Ledger:
    from .bibtex import parse

    claims = [
        Claim(id="C1", type=T.background, text="Potholes increase vehicle maintenance costs and cause road accidents", refs=["example2020cost"]),
        Claim(id="C2", type=T.background, text="Most municipalities detect potholes through manual inspection or citizen reports", refs=["example2021survey"]),
        Claim(id="C3", type=T.background, text="Camera-based detectors require clear visibility and dedicated hardware", refs=["example2022vision"]),
        Claim(id="C4", type=T.gap, text="No existing method detects potholes using only the accelerometer of an ordinary smartphone mounted on a two-wheeler"),
        Claim(id="C5", type=T.objective, text="We aim to detect potholes from smartphone accelerometer signals recorded on two-wheelers"),
        Claim(id="C6", type=T.contribution, text="We release a dataset of 1,240 labelled pothole events recorded on 85 km of urban roads"),
        Claim(id="C7", type=T.contribution, text="We propose a 1D convolutional network with 48,000 parameters that runs on the phone"),
        Claim(id="C8", type=T.dataset, text="Data were collected by 12 riders over 6 weeks using phones sampling at 100 Hz", tables=[]),
        Claim(id="C9", type=T.method, text="Signals were segmented into 2-second windows with 50% overlap"),
        Claim(id="C10", type=T.method, text="The network has three convolutional layers followed by global average pooling and a sigmoid output"),
        Claim(id="C11", type=T.method, text="We trained the network with the Adam optimiser at a learning rate of 0.001 for 40 epochs"),
        Claim(id="C12", type=T.method, text="We compared against a threshold detector and a random forest on hand-crafted features"),
        Claim(id="C13", type=T.result, text="The network reached an F1-score of 0.91 on the held-out test riders", tables=["T1"]),
        Claim(id="C14", type=T.comparison, text="The random forest reached an F1-score of 0.84 and the threshold detector 0.71", tables=["T1"]),
        Claim(id="C15", type=T.result, text="Inference took 3.2 ms per window on a mid-range phone", tables=["T1"]),
        Claim(id="C16", type=T.interpretation, text="We attribute the gain over the random forest to the network learning features from raw signals rather than relying on hand-crafted statistics"),
        Claim(id="C17", type=T.limitation, text="All data came from a single city with mostly asphalt roads"),
        Claim(id="C18", type=T.future_work, text="We plan to test the detector on concrete and unpaved roads"),
    ]
    tables = [DataTable(
        id="T1", caption="Detection performance on held-out riders",
        columns=["Method", "Precision", "Recall", "F1-score", "Latency (ms)"],
        rows=[
            ["Threshold detector", "0.66", "0.77", "0.71", "0.1"],
            ["Random forest", "0.86", "0.82", "0.84", "5.8"],
            ["1D CNN (ours)", "0.92", "0.90", "0.91", "3.2"],
        ],
    )]
    return Ledger(
        title="Pothole Detection from Smartphone Accelerometers on Two-Wheelers (Demo, fictional data)",
        authors=["Demo Author"],
        affiliations=["Example University"],
        keywords=["pothole detection", "accelerometer", "smartphone sensing", "1D CNN"],
        template="ieee",
        claims=claims,
        references=parse(DEMO_BIB),
        tables=tables,
        figures=[Figure(id="F1", caption="Accelerometer trace of a pothole event")],
    )
