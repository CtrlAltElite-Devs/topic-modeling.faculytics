"""
generate_report.py — Full tuning report for all runs (001–013).
Generates a PDF with executive summary, architecture, run log, metric progression,
topic inventory for key runs, and production baseline recommendation.
"""
import json
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

BASE = Path(__file__).parent.parent
EXPERIMENTS = BASE / "experiments"
REPORTS = BASE / "reports"
REPORTS.mkdir(exist_ok=True)

# ── Styles ──────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18, spaceAfter=12, textColor=colors.HexColor("#1a1a2e"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceAfter=8, spaceBefore=14, textColor=colors.HexColor("#16213e"))
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11, spaceAfter=6, spaceBefore=10, textColor=colors.HexColor("#0f3460"))
BODY = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.5, leading=14, spaceAfter=6, alignment=TA_JUSTIFY)
MONO = ParagraphStyle("Mono", parent=styles["Code"], fontSize=8.5, leading=12, spaceAfter=4,
                       backColor=colors.HexColor("#f4f4f4"), textColor=colors.HexColor("#333333"))
LABEL = ParagraphStyle("Label", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#555555"))
TITLE = ParagraphStyle("Title", parent=styles["Title"], fontSize=22, spaceAfter=6,
                        textColor=colors.HexColor("#1a1a2e"), alignment=TA_CENTER)
SUBTITLE = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=11, spaceAfter=20,
                           textColor=colors.HexColor("#555555"), alignment=TA_CENTER)
CAPTION = ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#777777"), alignment=TA_CENTER)

# ── Data ────────────────────────────────────────────────────────────────────
RUNS = {
    "001": {"dataset": "augmented (7,068)", "change": "Baseline — no preprocessing, default BERTopic params"},
    "002": {"dataset": "augmented (7,068)", "change": "Added nr_topics=20 to force merge"},
    "003": {"dataset": "augmented (7,068)", "change": "Reduced min_topic_size to 10"},
    "004": {"dataset": "augmented (7,068)", "change": "Added embedding_coherence metric; UMAP/HDBSCAN tuning grid start"},
    "005": {"dataset": "augmented (7,068)", "change": "Grid: min_topic_size=20, nr_topics=15"},
    "006": {"dataset": "augmented (7,068)", "change": "Grid: min_topic_size=10, nr_topics=None (auto)"},
    "007": {"dataset": "augmented (7,068)", "change": "Grid: min_topic_size=30, nr_topics=10"},
    "008": {"dataset": "augmented (7,068)", "change": "Added role stop words; fixed embedding coherence bug (LaBSE not loaded)"},
    "009": {"dataset": "augmented (7,068)", "change": "Added Cebuano possessives/fillers to stop list"},
    "010": {"dataset": "augmented (7,068)", "change": "Added English generic verbs/fillers — confirmed plateau"},
    "011": {"dataset": "augmented (7,068)", "change": "ARCHITECTURE PIVOT: KeyBERTInspired replaces c-TF-IDF — major quality jump"},
    "012": {"dataset": "real raw (18,559 / 34k)", "change": "Validated Run 011 params on real UC dataset (unfiltered)"},
    "013": {"dataset": "real filtered (12,297 / 34k)", "change": "Sentiment-gated filter: positive <10 words excluded; production baseline"},
}

def load_metrics(run_id):
    p = EXPERIMENTS / f"run_{run_id}" / "metrics.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}

def load_topics(run_id):
    p = EXPERIMENTS / f"run_{run_id}" / "topics.md"
    if p.exists():
        return p.read_text()
    return ""

def parse_topics(md_text):
    topics = []
    current = {}
    for line in md_text.splitlines():
        if line.startswith("## Topic"):
            if current:
                topics.append(current)
            label = line.split(": ", 1)[-1] if ": " in line else line
            current = {"label": label, "docs": 0, "keywords": ""}
        elif line.startswith("- Documents:"):
            current["docs"] = int(line.split(": ")[-1].strip())
        elif line.startswith("- Keywords:"):
            current["keywords"] = line.split(": ", 1)[-1].strip()
    if current:
        topics.append(current)
    return topics

def fmt(val, key):
    if val is None or val == "N/A":
        return "—"
    if key in ("embedding_coherence", "topic_diversity", "outlier_ratio", "silhouette_score", "npmi_coherence"):
        return f"{val:.4f}"
    if key == "num_topics":
        return str(int(val))
    return str(val)

def color_cell(val, key):
    if val is None:
        return colors.white
    if key == "embedding_coherence":
        return colors.HexColor("#d4edda") if val >= 0.5 else colors.HexColor("#fff3cd") if val >= 0.45 else colors.HexColor("#f8d7da")
    if key == "topic_diversity":
        return colors.HexColor("#d4edda") if val >= 0.7 else colors.HexColor("#f8d7da")
    if key == "outlier_ratio":
        return colors.HexColor("#d4edda") if val <= 0.2 else colors.HexColor("#fff3cd") if val <= 0.33 else colors.HexColor("#f8d7da")
    return colors.white

# ── Build ────────────────────────────────────────────────────────────────────
def build():
    out = REPORTS / "topic_modeling_full_report_runs_001_013.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4,
                             leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2.5*cm, bottomMargin=2.5*cm)
    story = []

    # ── Cover ──
    story += [
        Spacer(1, 2*cm),
        Paragraph("Faculytics 2.0", SUBTITLE),
        Paragraph("Topic Modeling Pipeline — Full Tuning Report", TITLE),
        Paragraph("Runs 001 – 013 | LaBSE + BERTopic + KeyBERTInspired", SUBTITLE),
        Spacer(1, 0.5*cm),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")),
        Spacer(1, 0.3*cm),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} PHT", CAPTION),
        Paragraph("Repo: github.com/CtrlAltElite-Devs/topic-modeling.faculytics", CAPTION),
        PageBreak(),
    ]

    # ── Executive Summary ──
    story += [
        Paragraph("Executive Summary", H1),
        Paragraph(
            "This report documents all 13 runs of the Faculytics 2.0 multilingual topic modeling pipeline, "
            "from initial baseline through architecture pivots to production baseline selection. "
            "The pipeline discovers educator-interpretable themes from student feedback written in "
            "English, Cebuano, Tagalog, Taglish, and code-switched variants.", BODY),
        Paragraph("<b>Final Architecture (Run 011 onwards):</b> Multilingual preprocessing → "
                  "LaBSE embeddings (Feng et al., 2022) → BERTopic with UMAP + HDBSCAN "
                  "(Grootendorst, 2022; McInnes et al., 2017) → KeyBERTInspired keyword extraction.", BODY),
        Paragraph("<b>Production Baseline (Run 013):</b> Sentiment-gated real dataset — "
                  "negative/neutral feedback always included; positive feedback included only if ≥10 words. "
                  "Applied to UC internal evaluation portal data (34,727 raw entries → 12,297 after gating and preprocessing).", BODY),
    ]

    # Targets table
    story += [Paragraph("Metric Targets vs Final Results", H3)]
    tdata = [
        ["Metric", "Target", "Run 011\n(augmented)", "Run 013\n(production)", "Status"],
        ["Embedding Coherence", "> 0.5", "0.5968", "0.6495", "✅ PASS"],
        ["Topic Diversity", "> 0.7", "0.9474", "0.9789", "✅ PASS"],
        ["Outlier Ratio", "< 20%", "33.0%", "52.0%", "⚠ Data-inherent"],
        ["Topic Count", "10–25", "19", "19", "✅ PASS"],
    ]
    t = Table(tdata, colWidths=[4*cm, 2.5*cm, 3*cm, 3*cm, 2.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("ALIGN", (1,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story += [t, Spacer(1, 0.3*cm),
              Paragraph("Note: Outlier ratio is data-inherent for HDBSCAN on short multilingual text "
                        "(consistent across all runs; accepted as thesis limitation).", LABEL),
              PageBreak()]

    # ── Pipeline Architecture ──
    story += [
        Paragraph("Pipeline Architecture", H1),
        Paragraph("1. Multilingual Preprocessing", H2),
        Paragraph("Raw feedback undergoes a multi-stage cleaning pipeline designed for short, informal, "
                  "code-switched text from Philippine students:", BODY),
        Paragraph("• Drop: Excel formula artifacts (#NAME?), entries shorter than 3 words post-cleaning, "
                  "pure keyboard gibberish", BODY),
        Paragraph("• Strip: broken Unicode (\ufffd), laughter noise (hahaha/hehe/lol), repeated characters "
                  "(everrr→ever), punctuation spam (???→.), URLs", BODY),
        Paragraph("• Multilingual stop words: English function words + Cebuano possessives "
                  "(iyang, kanyang, among, atong) + role words (propesor, estudyante, guro, maam, sir) "
                  "+ Tagalog function words", BODY),
        Paragraph("No spell correction is applied — LaBSE's WordPiece tokenization provides inherent "
                  "robustness to misspellings, and no Cebuano/Tagalog lexicons are available for "
                  "automated correction.", BODY),

        Paragraph("2. LaBSE Embedding (Feng et al., 2022)", H2),
        Paragraph("Language-agnostic BERT Sentence Embedding (LaBSE) encodes all feedback into a "
                  "shared 768-dimensional semantic space. Trained on 109 languages including low-resource "
                  "Philippine languages, LaBSE handles mid-sentence code-switching natively without "
                  "requiring language detection or translation preprocessing. "
                  "Embeddings are cached per dataset to avoid redundant GPU computation.", BODY),

        Paragraph("3. BERTopic Clustering (Grootendorst, 2022)", H2),
        Paragraph("BERTopic discovers topics via two sequential steps:", BODY),
        Paragraph("• <b>UMAP</b> (McInnes et al., 2020): Reduces 768-dim LaBSE embeddings to 10 dimensions "
                  "(n_neighbors=20, n_components=10, metric=cosine, random_state=42) preserving local "
                  "semantic structure", BODY),
        Paragraph("• <b>HDBSCAN</b> (McInnes et al., 2017): Density-based clustering "
                  "(min_cluster_size=15, min_samples=5, cluster_selection_method=eom). "
                  "Documents that don't fit any cluster are assigned to noise class (-1 / outliers). "
                  "nr_topics=20 triggers hierarchical merging to an educator-interpretable count.", BODY),

        Paragraph("4. KeyBERTInspired Keyword Extraction (Grootendorst, 2022 §4.2)", H2),
        Paragraph("Replaces the default c-TF-IDF keyword extraction. KeyBERTInspired selects keywords "
                  "by computing cosine similarity between candidate word embeddings and the topic "
                  "centroid in LaBSE space — finding words semantically closest to the cluster's "
                  "meaning rather than just the most frequent words. This is language-agnostic and "
                  "eliminates the stop-word treadmill observed in Runs 001–010 (where each run revealed "
                  "new frequency-dominant noise requiring manual stop-word addition).", BODY),

        Paragraph("5. Sentiment-Gated Pre-filtering (Production)", H2),
        Paragraph("Applied before topic modeling in production (Run 013 onwards). Since Faculytics "
                  "already runs sentiment analysis (Qwen 3B LoRA), the sentiment signal gates topic "
                  "modeling input:", BODY),
        Paragraph("• Negative feedback → always included (primary source of actionable insights)", BODY),
        Paragraph("• Neutral feedback → always included", BODY),
        Paragraph("• Positive feedback, ≥10 words → included (substantive positive feedback)", BODY),
        Paragraph("• Positive feedback, <10 words → excluded ('awesome teacher', 'good job', '10/10')", BODY),
        Paragraph("This filter removed 35.4% of real feedback (7,086 entries) that would otherwise "
                  "dominate topic clusters with generic praise noise.", BODY),
        PageBreak(),
    ]

    # ── Evaluation Metrics ──
    story += [
        Paragraph("Evaluation Metrics", H1),
        Paragraph("Primary: Embedding Coherence", H2),
        Paragraph("Average pairwise cosine similarity between LaBSE embeddings of each topic's "
                  "top-10 keywords. Higher values indicate that the keywords of a topic are "
                  "semantically close together in the shared embedding space — i.e., they genuinely "
                  "belong to the same concept. Target: > 0.5.", BODY),
        Paragraph("This metric was chosen over NPMI because NPMI assumes monolingual co-occurrence: "
                  "semantically equivalent words in different languages (e.g., 'teacher' and 'guro') "
                  "never appear in the same document, producing systematically negative NPMI scores "
                  "that are a structural artifact of multilingual data, not a quality signal "
                  "(Hoyle et al., 2021; Lau et al., 2014).", BODY),
        Paragraph("Secondary: Topic Diversity", H2),
        Paragraph("Ratio of unique words across all topics' top-10 keyword lists. Low diversity "
                  "indicates overlapping or redundant topics. Target: > 0.7.", BODY),
        Paragraph("Reference Only: NPMI Coherence", H2),
        Paragraph("Retained for completeness but not used as primary metric. Consistently 0.0 or "
                  "negative across all multilingual runs — confirmed structural artifact.", BODY),
        Paragraph("Outlier Ratio", H2),
        Paragraph("Proportion of documents assigned to HDBSCAN noise cluster (-1). Consistent at "
                  "~33% on augmented data across all runs regardless of parameter variation. "
                  "Higher on real data (47–52%) due to greater linguistic diversity and shorter "
                  "average entry length. Accepted as data-inherent and documented as thesis limitation.", BODY),
        PageBreak(),
    ]

    # ── Full Run Log ──
    story += [Paragraph("Run-by-Run Log", H1)]

    run_notes = {
        "001": "Baseline. 78 topics — far too many. No stop words. NPMI used as sole metric. "
               "c-TF-IDF keywords dominated by generic noise (maam, sir, teacher).",
        "002": "Added nr_topics=20 to force hierarchical merge. Down to 38 topics — improvement. "
               "Still high outlier rate. NPMI went negative — first sign of multilingual metric problem.",
        "003": "Reduced min_topic_size to 10. Topics jumped to 82 — too granular. "
               "Confirmed nr_topics constraint is necessary.",
        "004": "Introduced embedding_coherence as primary metric. Added UMAP/HDBSCAN grid search. "
               "First clean 19-topic result. Composite score formula established.",
        "005": "Grid point: larger min_topic_size (20), fewer topics (15). Topic count dropped "
               "below interpretable range. Coherence slightly higher but diversity dropped.",
        "006": "Grid: small clusters (min_topic_size=10), auto topic count (nr_topics=None). "
               "52 topics — too many for educators. Composite score penalized.",
        "007": "Grid: large clusters (min_topic_size=30), very few topics (10). Below minimum "
               "target of 10 interpretable topics. Coherence plateau confirmed.",
        "008": "Added role words to stop list. Fixed critical bug: LaBSE not loaded in evaluate.py "
               "→ embedding_coherence was computing random vector similarities = 0.0. "
               "Bug masked all coherence improvements since Run 004.",
        "009": "Added Cebuano possessives and fillers to stop word list. First valid coherence reading "
               "post-bug-fix: 0.4826. 11/19 topics clearly interpretable. "
               "Identified stop-word treadmill problem: c-TF-IDF is architecturally frequency-based.",
        "010": "Added English generic verbs. Metrics identical to Run 009 — confirmed plateau. "
               "Stop-word expansion no longer effective. Architectural change required.",
        "011": "ARCHITECTURE PIVOT: Replaced c-TF-IDF with KeyBERTInspired representation model. "
               "Coherence: 0.4826→0.5968 (+23.7%). Diversity: 0.9053→0.9474. "
               "18/19 topics clearly interpretable. First run to pass coherence target. "
               "AUGMENTED DATASET BASELINE LOCKED.",
        "012": "Validated Run 011 architecture on real UC dataset (unfiltered, 34k→18.5k after preprocessing). "
               "Coherence and diversity higher than augmented, but outlier rate jumped to 47.8%. "
               "Topic quality degraded: Topics 0+1 near-duplicate 'good teaching' clusters, "
               "Christmas greetings topic, '10/10' rating topic, slang clusters. "
               "Root cause: real feedback is overwhelmingly short positive praise noise.",
        "013": "Applied sentiment-gated pre-filter: positive <10 words excluded (35.4% of real data). "
               "Filtered dataset: 12,297 docs. New topics emerged: asynchronous attendance (Topic 5), "
               "specific tardiness timing (Topic 12). Outlier rate increased to 52% "
               "(filter removed easy-to-cluster positive entries). PRODUCTION BASELINE SELECTED.",
    }

    for run_id, info in RUNS.items():
        m = load_metrics(run_id)
        story += [
            Paragraph(f"Run {run_id} — {info['change']}", H2),
            Paragraph(f"Dataset: {info['dataset']}", LABEL),
            Spacer(1, 0.15*cm),
        ]
        # Metrics row
        keys = ["embedding_coherence", "topic_diversity", "outlier_ratio", "num_topics"]
        mrow = [["Embed Coherence", "Topic Diversity", "Outlier Ratio", "Num Topics"]]
        vrow = [[fmt(m.get(k), k) for k in keys]]
        mt = Table(mrow + vrow, colWidths=[3.8*cm]*4)
        mt.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e8ecf0")),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            *([("BACKGROUND", (i,1), (i,1), color_cell(m.get(keys[i]), keys[i]))
               for i in range(len(keys)) if m.get(keys[i]) is not None])
        ]))
        story += [mt, Spacer(1, 0.15*cm)]
        story += [Paragraph(run_notes.get(run_id, ""), BODY), Spacer(1, 0.2*cm)]

    story.append(PageBreak())

    # ── Metric Progression Table ──
    story += [Paragraph("Metric Progression — All Runs", H1)]
    headers = ["Run", "Dataset", "Embed\nCoherence", "Topic\nDiversity", "Outlier\nRatio", "Num\nTopics"]
    tdata = [headers]
    for run_id in RUNS:
        m = load_metrics(run_id)
        ds = "augmented" if "augmented" in RUNS[run_id]["dataset"] else ("real-filtered" if "filtered" in RUNS[run_id]["dataset"] else "real-raw")
        tdata.append([
            run_id,
            ds,
            fmt(m.get("embedding_coherence"), "embedding_coherence"),
            fmt(m.get("topic_diversity"), "topic_diversity"),
            fmt(m.get("outlier_ratio"), "outlier_ratio"),
            fmt(m.get("num_topics"), "num_topics"),
        ])
    pt = Table(tdata, colWidths=[1.2*cm, 2.8*cm, 2.8*cm, 2.8*cm, 2.5*cm, 2*cm])
    style = [
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        # Highlight Run 011 and 013
        ("BACKGROUND", (0,11), (-1,11), colors.HexColor("#d4edda")),
        ("BACKGROUND", (0,13), (-1,13), colors.HexColor("#cce5ff")),
        ("FONTNAME", (0,11), (-1,11), "Helvetica-Bold"),
        ("FONTNAME", (0,13), (-1,13), "Helvetica-Bold"),
    ]
    pt.setStyle(TableStyle(style))
    story += [pt,
              Spacer(1, 0.2*cm),
              Paragraph("Green = augmented dataset baseline (Run 011). Blue = production baseline (Run 013).", LABEL),
              PageBreak()]

    # ── Topic Inventory ── Run 011 and Run 013
    for run_id, label in [("011", "Run 011 — Augmented Dataset Baseline (KeyBERTInspired)"),
                           ("013", "Run 013 — Production Baseline (Sentiment-Gated Real Data)")]:
        story += [Paragraph(f"Topic Inventory: {label}", H1)]
        topics = parse_topics(load_topics(run_id))
        tdata = [["#", "Topic Label", "Docs", "Top Keywords"]]
        for t in topics:
            num = t["label"].split("_")[0] if "_" in t["label"] else "?"
            kw = t["keywords"][:80] + "..." if len(t["keywords"]) > 80 else t["keywords"]
            tdata.append([num, t["label"].split(": ", 1)[-1][:50] if ": " in t["label"] else t["label"][:50], str(t["docs"]), kw])
        tt = Table(tdata, colWidths=[0.8*cm, 4*cm, 1.5*cm, 8.7*cm])
        tt.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#16213e")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f9fa")]),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("WORDWRAP", (3,1), (3,-1), True),
        ]))
        story += [tt, Spacer(1, 0.3*cm)]
        if run_id == "011":
            story.append(PageBreak())

    story.append(PageBreak())

    # ── Production Recommendation ──
    story += [
        Paragraph("Production Baseline Recommendation", H1),
        Paragraph("Based on 13 experimental runs, the following configuration is recommended for "
                  "production deployment in the Faculytics 2.0 worker:", BODY),
    ]
    rec = [
        ["Component", "Setting", "Rationale"],
        ["Embedding model", "LaBSE (sentence-transformers/LaBSE)", "109-language support, handles code-switching natively"],
        ["Preprocessing", "Multilingual stop words + noise filters", "Removes Excel artifacts, role words, Cebuano/Tagalog fillers"],
        ["Sentiment gate", "Positive <10 words → excluded", "Removes generic praise noise (35.4% of real feedback)"],
        ["UMAP", "n_neighbors=20, n_components=10, cosine", "Preserves local semantic structure at scale"],
        ["HDBSCAN", "min_cluster_size=15, min_samples=5, eom", "Stable cluster detection; eom avoids leaf-level over-splitting"],
        ["Topic count", "nr_topics=20", "Hierarchical merge to educator-interpretable count"],
        ["Keywords", "KeyBERTInspired (LaBSE-backed)", "Language-agnostic; immune to frequency-based stop-word treadmill"],
        ["Primary metric", "Embedding coherence > 0.5", "Valid for multilingual data; NPMI is not"],
        ["Min feedback", "30 responses per faculty/semester", "Below threshold: sentiment-only, no topic modeling"],
    ]
    rt = Table(rec, colWidths=[3.5*cm, 5.5*cm, 6*cm])
    rt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story += [rt, Spacer(1, 0.4*cm)]

    # Known limitations
    story += [
        Paragraph("Known Limitations", H2),
        Paragraph("• <b>Outlier rate</b> (~33–52%): Consistent across all parameter combinations on both datasets. "
                  "Data-inherent for short multilingual text with HDBSCAN. Outlier documents still receive "
                  "sentiment analysis.", BODY),
        Paragraph("• <b>Positive long feedback</b> clusters (Topics 0, 1 in Run 013): Long positive feedback "
                  "still generates 'good teaching' clusters. Acceptable as dashboard insight "
                  "('majority of feedback is positive about teaching quality'). Handle at UI layer — "
                  "do not surface these clusters as 'problems'.", BODY),
        Paragraph("• <b>Seasonal noise</b> (Christmas greetings, Topic 13): ~42 entries survived the word "
                  "count filter by being verbose. Low document count — acceptable.", BODY),
        Paragraph("• <b>Feedback source</b>: Data is from the school's internal evaluation portal, not Moodle. "
                  "Moodle provides course/student metadata; feedback data is separate.", BODY),
        Paragraph("• <b>No spell correction</b>: No Cebuano/Tagalog lexicons available. LaBSE WordPiece "
                  "tokenization provides robustness. Defensible limitation for thesis.", BODY),
    ]

    # Citations
    story += [
        Paragraph("References", H2),
        Paragraph("Feng, F., Yang, Y., Cer, D., Arivazhagan, N., & Wang, W. (2022). Language-agnostic BERT sentence embedding. "
                  "arXiv:2007.01852.", BODY),
        Paragraph("Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. "
                  "arXiv:2203.05794.", BODY),
        Paragraph("Hoyle, A., Goel, P., Resnik, A., & Resnik, P. (2021). Is automated topic model evaluation broken? "
                  "The incoherence of coherence. NeurIPS 2021.", BODY),
        Paragraph("Lau, J. H., Newman, D., & Baldwin, T. (2014). Machine reading tea leaves: Automatically evaluating "
                  "topic coherence and topic model quality. EACL 2014.", BODY),
        Paragraph("McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical density based clustering. "
                  "Journal of Open Source Software, 2(11), 205.", BODY),
        Paragraph("McInnes, L., Healy, J., & Melville, J. (2020). UMAP: Uniform manifold approximation and projection "
                  "for dimension reduction. arXiv:1802.03426.", BODY),
    ]

    doc.build(story)
    print(f"✅ Report generated: {out}")
    print(f"   Size: {out.stat().st_size // 1024} KB")
    return out

if __name__ == "__main__":
    build()
