# Notebooks

Interactive notebooks for exploring and analyzing the synthetic dataset.

## Synthetic Data Explorer

An interactive marimo notebook for exploring the synthetic dataset generated for Gemma3-270M fine-tuning.

### Features

- 📊 **Dataset Overview**: Statistics about train/test splits, lengths, and distributions
- 📏 **Request Length Analysis**: Interactive histogram with filtering
- 🌿 **Branch Prefix Distribution**: Top 20 most common branch prefixes
- 📝 **Summary Analysis**: Word count distribution and statistics
- 🔍 **Text Pattern Analysis**: Detects programming languages and app types mentioned
- 📋 **Sample Examples**: Browse random examples with refresh
- ✅ **Data Quality Checks**: Identifies potential issues
- 💾 **Export Options**: Export filtered data to CSV/JSON/JSONL

### Installation

Install marimo if you haven't already:

```bash
# Using uv (recommended)
uv add marimo altair pandas

# Or using pip
pip install marimo altair pandas
```

### Running the Notebook

#### Option 1: Edit Mode (Interactive)

```bash
# From the project root
uv run marimo edit notebooks/explore_synthetic_data.py

# Or if marimo is already installed
marimo edit notebooks/explore_synthetic_data.py
```

This opens the notebook in your browser with full editing and interaction capabilities.

#### Option 2: Run Mode (View Only)

```bash
uv run marimo run notebooks/explore_synthetic_data.py
```

This runs the notebook as a read-only app.

#### Option 3: Export to HTML

```bash
# Export as static HTML
uv run marimo export html notebooks/explore_synthetic_data.py -o synthetic_data_report.html

# Export as interactive HTML (with code)
uv run marimo export html notebooks/explore_synthetic_data.py --include-code -o synthetic_data_report.html
```

### Usage Tips

1. **Dataset Selection**: Use the dropdown to select train, test, or both datasets
2. **Interactive Filtering**: Click and drag on the request length chart to filter examples
3. **Sample Browsing**: Adjust the slider and click refresh to see different examples
4. **Export**: Use the export section to download filtered data for further analysis

### What the Notebook Shows

#### Dataset Overview
- Number of train/test examples
- Average request length
- Average summary word count
- Number of unique branch prefixes

#### Request Length Distribution
- Histogram showing the distribution of request lengths
- Interactive filtering by clicking and dragging
- Identifies short, medium, and long requests

#### Branch Prefix Distribution
- Bar chart of top 20 branch prefixes (bug/, feat/, etc.)
- Color-coded for easy identification
- Shows which types of changes are most common

#### Summary Analysis
- Distribution of summary word counts
- Statistics: min, max, mean, median
- Identifies summaries outside the 2-4 word range

#### Text Pattern Analysis
- Detects mentions of programming languages (Python, Go, Rust, etc.)
- Detects mentions of application types (Web App, CLI, API, etc.)
- Useful for understanding diversity

#### Sample Examples
- Browse random examples from the dataset
- Shows full request (or truncated if long)
- Shows summary and branch name
- Refresh button for new samples

#### Data Quality Checks
- Empty summaries/branches
- Summaries with < 2 or > 4 words
- Branches missing category prefix (bug/, feat/, etc.)
- Helps identify labeling issues

## Keyboard Shortcuts (Edit Mode)

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Run cell |
| `Shift+Enter` | Run cell and move to next |
| `Ctrl+S` | Save notebook |
| `Ctrl+Shift+E` | Export as script |

## Dependencies

The notebook requires:
- `marimo` - Reactive notebook framework
- `pandas` - Data manipulation
- `altair` - Interactive visualizations
- `json` - JSON parsing (standard library)
- `pathlib` - Path handling (standard library)
- `collections` - Counter (standard library)
- `re` - Regular expressions (standard library)

## Troubleshooting

### Issue: "Module not found: marimo"

**Solution**: Install marimo
```bash
uv add marimo
```

### Issue: "Module not found: altair"

**Solution**: Install altair
```bash
uv add altair
```

### Issue: "File not found: data/synthetic/train.jsonl"

**Solution**: Run the data generation script first:
```bash
uv run python scripts/data/generate_enhanced_dataset.py --num-synthetic 2000
```

### Issue: Charts not rendering

**Solution**: Make sure you're running in edit or run mode (not as a regular Python script):
```bash
marimo edit notebooks/explore_synthetic_data.py
```

## Extending the Notebook

The notebook is modular and easy to extend. Some ideas:

1. **Add more visualizations**:
   - Word clouds of common terms
   - Network graph of related summaries
   - Time series if you track generation time

2. **Add more quality checks**:
   - Check for duplicate requests
   - Validate JSON format
   - Check for invalid characters

3. **Add model evaluation**:
   - Load fine-tuned model
   - Generate predictions
   - Compare with ground truth

4. **Add SQL queries**:
   - marimo has built-in SQL support
   - Query the dataset with SQL
   - Join with other data sources

## Resources

- [marimo Documentation](https://docs.marimo.io/)
- [Altair Documentation](https://altair-viz.github.io/)
- [marimo Examples](https://github.com/marimo-team/marimo/tree/main/examples)
- [marimo Discord](https://discord.gg/JE7nhX6mD8)

## Contributing

To add a new notebook:
1. Create a new `.py` file in the `notebooks/` directory
2. Use `marimo.App()` to define your notebook
3. Add documentation to this README
4. Test with `marimo edit notebooks/your_notebook.py`
