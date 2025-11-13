"""
Synthetic Data Explorer

An interactive notebook for exploring the synthetic dataset generated for Gemma3-270M fine-tuning.

Features:
- Overview statistics
- Interactive visualizations
- Data quality checks
- Sample inspection
- Distribution analysis
"""

import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import json
    import pandas as pd
    import altair as alt
    from pathlib import Path
    from collections import Counter
    import re
    return Counter, Path, alt, json, mo, pd, re


@app.cell
def _(mo):
    mo.md("""
    # 🔬 Synthetic Data Explorer

    Explore the synthetic dataset generated for Gemma3-270M fine-tuning.
    Use the controls below to filter and analyze the data.
    """)
    return


@app.cell
def _():
    import os
    os.getcwd()
    return


@app.cell
def _(Path):
    # Data paths
    DATA_DIR = Path("./data/synthetic")
    TRAIN_FILE = DATA_DIR / "train.jsonl"
    TEST_FILE = DATA_DIR / "test.jsonl"
    PROMPTS_FILE = DATA_DIR / "synthetic_prompts.jsonl"
    return TEST_FILE, TRAIN_FILE


@app.cell
def _(mo):
    # File selector
    dataset_selector = mo.ui.dropdown(
        options=["train", "test", "both"],
        value="both",
        label="Select Dataset:"
    )
    return (dataset_selector,)


@app.cell
def _(dataset_selector, mo):
    mo.md(f"""
    ## Dataset Selection\n\n{dataset_selector}
    """)
    return


@app.cell
def _(TEST_FILE, TRAIN_FILE, dataset_selector, json, pd):
    # Load data based on selection
    def load_dataset(file_path):
        data = []
        try:
            with open(file_path) as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        except FileNotFoundError:
            return []
        return data

    def parse_example(example):
        """Parse a training example into components"""
        conversations = example.get("conversations", [])
        if len(conversations) >= 2:
            user_content = conversations[0]["content"]
            assistant_content = conversations[1]["content"]

            # Extract request (after "Request:\n")
            request = ""
            if "Request:\n" in user_content:
                request = user_content.split("Request:\n", 1)[1]

            # Parse JSON response
            try:
                response = json.loads(assistant_content)
                summary = response.get("summary", "")
                branch = response.get("branch", "")
            except json.JSONDecodeError:
                summary = ""
                branch = ""

            return {
                "request": request,
                "request_length": len(request),
                "summary": summary,
                "branch": branch,
                "branch_prefix": branch.split("/")[0] if "/" in branch else "",
                "summary_word_count": len(summary.split())
            }
        return None

    # Load selected dataset
    train_data = []
    test_data = []

    if dataset_selector.value in ["train", "both"]:
        train_data = load_dataset(TRAIN_FILE)
    if dataset_selector.value in ["test", "both"]:
        test_data = load_dataset(TEST_FILE)

    # Combine and parse
    all_data = train_data + test_data
    parsed_data = [parse_example(ex) for ex in all_data if parse_example(ex)]

    # Create DataFrame
    df = pd.DataFrame(parsed_data)
    df[["request","summary","branch"]]
    return df, test_data, train_data


@app.cell
def _(TEST_FILE, TRAIN_FILE, df, mo, test_data, train_data):
    # Overview Statistics
    mo.md(
        f"""
        ## 📊 Dataset Overview

        | Metric | Value |
        |--------|-------|
        | **Train Examples** | {len(train_data)} |
        | **Test Examples** | {len(test_data)} |
        | **Total Examples** | {len(df)} |
        | **Avg Request Length** | {df['request_length'].mean():.0f} chars |
        | **Avg Summary Words** | {df['summary_word_count'].mean():.1f} words |
        | **Unique Branch Prefixes** | {df['branch_prefix'].nunique()} |

        **Data Files:**
        - Train: `{TRAIN_FILE}`
        - Test: `{TEST_FILE}`
        """
    )
    return


@app.cell
def _(alt, df, mo):
    # Request Length Distribution
    length_chart = mo.ui.altair_chart(
        alt.Chart(df).mark_bar().encode(
            x=alt.X('request_length:Q', bin=alt.Bin(maxbins=50), title='Request Length (characters)'),
            y=alt.Y('count()', title='Count'),
            tooltip=['count()']
        ).properties(
            title='Request Length Distribution',
            width=700,
            height=300
        ).interactive()
    )
    return (length_chart,)


@app.cell
def _(length_chart, mo):
    mo.md(f"""
    ## 📏 Request Length Analysis

    {length_chart}

    Click and drag on the chart to filter examples by length.
    """)
    return


@app.cell
def _(df, length_chart):
    # Filter based on chart selection
    if length_chart.value is not None and not length_chart.value.empty:
        filtered_df = length_chart.value
    else:
        filtered_df = df
    return (filtered_df,)


@app.cell
def _(Counter, filtered_df, mo):
    # Branch Prefix Distribution
    prefix_counts = Counter(filtered_df['branch_prefix'])
    prefix_data = [{"prefix": k, "count": v} for k, v in prefix_counts.most_common(20)]

    mo.md(
        f"""
        ## 🌿 Branch Prefix Distribution

        Top 20 most common branch prefixes in the filtered data.
        """
    )
    return (prefix_data,)


@app.cell
def _(alt, mo, pd, prefix_data):
    prefix_df = pd.DataFrame(prefix_data)

    prefix_chart = mo.ui.altair_chart(
        alt.Chart(prefix_df).mark_bar().encode(
            x=alt.X('count:Q', title='Count'),
            y=alt.Y('prefix:N', sort='-x', title='Branch Prefix'),
            tooltip=['prefix', 'count'],
            color=alt.Color('prefix:N', legend=None)
        ).properties(
            width=700,
            height=400
        ).interactive()
    )

    prefix_chart
    return


@app.cell
def _(filtered_df, mo):
    # Summary Word Count Distribution
    mo.md(
        f"""
        ## 📝 Summary Analysis

        Summary statistics for the filtered examples:
        - Min words: {filtered_df['summary_word_count'].min():.0f}
        - Max words: {filtered_df['summary_word_count'].max():.0f}
        - Mean words: {filtered_df['summary_word_count'].mean():.1f}
        - Median words: {filtered_df['summary_word_count'].median():.1f}
        """
    )
    return


@app.cell
def _(alt, filtered_df, mo):
    summary_chart = mo.ui.altair_chart(
        alt.Chart(filtered_df).mark_bar().encode(
            x=alt.X('summary_word_count:Q', bin=alt.Bin(maxbins=10), title='Summary Word Count'),
            y=alt.Y('count()', title='Count'),
            tooltip=['count()']
        ).properties(
            title='Summary Word Count Distribution',
            width=700,
            height=300
        ).interactive()
    )

    summary_chart
    return


@app.cell
def _(mo):
    # Text Analysis
    mo.md("""
    ## 🔍 Text Pattern Analysis

    Analyzing common patterns in requests.
    """)
    return


@app.cell
def _(Counter, filtered_df, re):
    # Detect programming languages mentioned
    languages = ['python', 'javascript', 'typescript', 'go', 'rust', 'ruby', 'java', 'c\\+\\+', 'zig']

    def detect_languages(text):
        found = []
        for lang in languages:
            if re.search(rf'\b{lang}\b', text.lower()):
                found.append(lang.replace('\\+\\+', '++'))
        return found

    all_languages = []
    for _req in filtered_df['request']:
        all_languages.extend(detect_languages(_req))

    lang_counts = Counter(all_languages)
    return (lang_counts,)


@app.cell
def _(lang_counts, mo, pd):
    lang_data = [{"language": k, "mentions": v} for k, v in lang_counts.most_common()]
    lang_df = pd.DataFrame(lang_data)

    if not lang_df.empty:
        mo.md(
            f"""
            ### Programming Languages Mentioned

            {mo.ui.table(lang_df)}
            """
        )
    else:
        mo.md("No programming languages detected in requests.")
    return


@app.cell
def _(Counter, filtered_df, re):
    # Detect application types
    app_types = ['web app', 'mobile app', 'desktop app', 'cli', 'api', 'library', 'sdk', 'microservice']

    def detect_app_types(text):
        found = []
        for app in app_types:
            if re.search(rf'\b{app}\b', text.lower()):
                found.append(app)
        return found

    all_app_types = []
    for _req in filtered_df['request']:
        all_app_types.extend(detect_app_types(_req))

    app_type_counts = Counter(all_app_types)
    return (app_type_counts,)


@app.cell
def _(app_type_counts, mo, pd):
    app_type_data = [{"type": k, "mentions": v} for k, v in app_type_counts.most_common()]
    app_type_df = pd.DataFrame(app_type_data)

    if not app_type_df.empty:
        mo.md(
            f"""
            ### Application Types Mentioned

            {mo.ui.table(app_type_df)}
            """
        )
    else:
        mo.md("No application types detected in requests.")
    return


@app.cell
def _(mo):
    # Sample Examples
    mo.md("""
    ## 📋 Sample Examples

    Browse random examples from the filtered dataset.
    """)
    return


@app.cell
def _(mo):
    sample_size_slider = mo.ui.slider(
        start=1,
        stop=20,
        step=1,
        value=5,
        label="Number of samples:",
        show_value=True
    )

    refresh_button = mo.ui.button(label="🔄 Refresh Samples")
    return refresh_button, sample_size_slider


@app.cell
def _(mo, refresh_button, sample_size_slider):
    mo.hstack([sample_size_slider, refresh_button], justify="start")
    return


@app.cell
def _(filtered_df, mo, refresh_button, sample_size_slider):
    # Trigger refresh
    _refresh_trigger = refresh_button.value

    # Sample examples
    sample_df = filtered_df.sample(n=min(sample_size_slider.value, len(filtered_df))).reset_index(drop=True)

    # Display samples
    sample_cards = []
    for idx, row in sample_df.iterrows():
        request_preview = row['request'][:300] + "..." if len(row['request']) > 300 else row['request']

        card = mo.md(
            f"""
            **Example {idx + 1}**

            **Request** ({row['request_length']} chars):
            ```
            {request_preview}
            ```

            **Summary**: `{row['summary']}`
            **Branch**: `{row['branch']}`

            ---
            """
        )
        sample_cards.append(card)

    mo.vstack(sample_cards)
    return


@app.cell
def _(mo):
    # Data Quality Checks
    mo.md("""
    ## ✅ Data Quality Checks

    Checking for potential issues in the dataset.
    """)
    return


@app.cell
def _(filtered_df, mo):
    # Quality metrics
    empty_summaries = (filtered_df['summary'] == "").sum()
    empty_branches = (filtered_df['branch'] == "").sum()
    short_summaries = (filtered_df['summary_word_count'] < 2).sum()
    long_summaries = (filtered_df['summary_word_count'] > 4).sum()
    invalid_branches = filtered_df[~filtered_df['branch'].str.contains('/')].shape[0]

    quality_issues = []

    if empty_summaries > 0:
        quality_issues.append(f"⚠️ {empty_summaries} examples with empty summaries")
    if empty_branches > 0:
        quality_issues.append(f"⚠️ {empty_branches} examples with empty branches")
    if short_summaries > 0:
        quality_issues.append(f"⚠️ {short_summaries} examples with summaries < 2 words")
    if long_summaries > 0:
        quality_issues.append(f"⚠️ {long_summaries} examples with summaries > 4 words")
    if invalid_branches > 0:
        quality_issues.append(f"⚠️ {invalid_branches} examples with branches missing '/' prefix")

    if quality_issues:
        mo.md("\n".join(["### Issues Found:"] + quality_issues))
    else:
        mo.md("### ✅ No quality issues detected!")
    mo.md("\n".join(["### Issues Found:"] + quality_issues))
    return


@app.cell
def _(mo):
    # Export Options
    mo.md("""
    ## 💾 Export Options

    Export filtered data for further analysis.
    """)
    return


@app.cell
def _(mo):
    export_format = mo.ui.dropdown(
        options=["CSV", "JSON", "JSONL"],
        value="CSV",
        label="Export Format:"
    )

    export_button = mo.ui.button(label="📥 Export Filtered Data")
    return export_button, export_format


@app.cell
def _(export_button, export_format, mo):
    mo.hstack([export_format, export_button], justify="start")
    return


@app.cell
def _(export_button, export_format, filtered_df, mo):
    # Handle export
    if export_button.value:
        if export_format.value == "CSV":
            csv_data = filtered_df.to_csv(index=False)
            mo.download(csv_data, filename="filtered_data.csv")
            mo.md("✅ CSV export ready!")
        elif export_format.value == "JSON":
            json_data = filtered_df.to_json(orient="records", indent=2)
            mo.download(json_data, filename="filtered_data.json")
            mo.md("✅ JSON export ready!")
        elif export_format.value == "JSONL":
            jsonl_data = "\n".join(filtered_df.to_json(orient="records", lines=True).split("\n"))
            mo.download(jsonl_data, filename="filtered_data.jsonl")
            mo.md("✅ JSONL export ready!")
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    ## 📚 Documentation

    This notebook analyzes the synthetic dataset generated for Gemma3-270M fine-tuning.

    **Features:**
    - Interactive filtering by request length
    - Branch prefix distribution analysis
    - Programming language and app type detection
    - Data quality checks
    - Sample browsing
    - Export capabilities

    **Tips:**
    - Click and drag on charts to filter data
    - Use the refresh button to see different samples
    - Export filtered data for detailed analysis
    """)
    return


if __name__ == "__main__":
    app.run()
