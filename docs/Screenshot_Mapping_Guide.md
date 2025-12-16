# Screenshot Mapping Guide: LaTeX Report ↔ Jupyter Notebook

This guide maps each screenshot placeholder in `Bone_Age_Report.tex` to the corresponding cell in `Bone_Age_Prediction_Xception_FINAL.ipynb`.

---

## 📊 Figure 1: Predicted vs. True Ages (LaTeX Line ~175)
**Location in LaTeX:** Section 3.2 "Plot of Predicted vs. True Ages"  
**Notebook Cell:** **Cell 37** (markdown: "5.3 Prediction Visualization")  
**Actual Code Cell:** **Cell 38**  
**What to capture:** 
- Left plot: Scatter plot with predicted vs true bone ages
- Shows diagonal red "Perfect Prediction" line
- Title includes R² = 0.9169, MAE = 9.04 months

**How to capture:**
1. Run Cell 38 in your notebook
2. Screenshot the LEFT plot (scatter plot)
3. Crop to show just the plot with title and axes

---

## 📊 Figure 2: Residual Plot (LaTeX Line ~187)
**Location in LaTeX:** Section 3.2 "Plot of Predicted vs. True Ages"  
**Notebook Cell:** **Cell 38** (same cell as Figure 1)  
**What to capture:**
- Right plot: Residual plot (True - Predicted vs Predicted)
- Shows horizontal lines at 0 and ±12 months
- Title: "Residual Plot"

**How to capture:**
1. From the same Cell 38 output
2. Screenshot the RIGHT plot (residual plot)
3. Crop to show just the plot with title and axes

---

## 📈 Figure 3: Training History (LaTeX Line ~199)
**Location in LaTeX:** Section 3.3 "Training History"  
**Notebook Cell:** **Cell 27** (markdown: "5.1 Training History Visualization")  
**Actual Code Cell:** **Cell 28**  
**What to capture:**
- Both plots showing training/validation Loss and MAE curves
- Shows convergence over epochs
- Early stopping behavior visible

**How to capture:**
1. Run Cell 28 in your notebook
2. Screenshot BOTH plots (Loss and MAE side by side)
3. Include the text output below showing "Training Summary" stats

---

## ⚖️ Figure 4: Gender-Wise Analysis (LaTeX Line ~228)
**Location in LaTeX:** Section 3.4 "Gender-Wise Performance Analysis"  
**Notebook Cell:** **Cell 31** (markdown: "5.3 Gender-Wise Performance Analysis")  
**Actual Code Cell:** **Cell 32**  
**What to capture:**
- Both plots: Box plot (left) and Scatter plot by gender (right)
- Shows error distribution by gender
- Male vs Female comparison with different colors

**How to capture:**
1. Run Cell 32 in your notebook
2. Screenshot BOTH plots together
3. Include the text output showing gender-wise metrics (MAE, RMSE, R² for both genders)

---

## 🎯 Figure 5: Confusion Matrix (LaTeX Line ~252)
**Location in LaTeX:** Section 3.5 "Classification Model - Developmental Stages"  
**Notebook Cell:** **Cell 33** (markdown: "5.4 Classification Model")  
**Actual Code Cell:** **Cell 34**  
**What to capture:**
- Both plots: Confusion Matrix heatmap (left) and Stage Distribution bar chart (right)
- Shows Child/Adolescent/Adult classification
- Includes accuracy = 91.54%, QWK = 0.8248

**How to capture:**x
1. Run Cell 34 in your notebook
2. Screenshot BOTH plots together
3. Also capture the classification report text output above the plots

---

## 🔥 Figure 6: Grad-CAM Visualization (LaTeX Line ~264)
**Location in LaTeX:** Section 3.6 "Grad-CAM Visualization"  
**Notebook Cell:** **Cell 35** (markdown: "5.5 Grad-CAM Visualization")  
**Actual Code Cell:** **Cell 36**  
**What to capture:**
- Multiple rows showing Original Image | Heatmap | Overlay
- Shows 6 samples across error percentiles (10th, 30th, 50th, 70th, 90th, 95th)
- Each row includes gender, true age, predicted age, and error

**How to capture:**
1. Run Cell 36 in your notebook (takes time to generate)
2. Screenshot the ENTIRE figure with all 6 rows
3. This will be a tall image - that's expected
4. Alternatively, screenshot just 2-3 representative rows if size is too large

---

## 🔍 Figure 7: Difficult Samples (LaTeX Line ~289)
**Location in LaTeX:** Section 4.1 "Interpretation of Errors and Difficult Samples"  
**Notebook Cell:** **Cell 39** (markdown: "5.4 Sample Predictions")  
**Actual Code Cell:** **Cell 40**  
**What to capture:**
- 6 X-ray images in 2 rows × 3 columns grid
- Each shows: Gender, True age, Predicted age, Error
- Green titles = good predictions (≤12 months error)
- Red titles = poor predictions (>12 months error)

**How to capture:**
1. Run Cell 40 in your notebook
2. Screenshot the entire 2×3 grid of sample predictions
3. Make sure the colored titles (green/red) are visible

---

## 📋 Quick Reference Table

| LaTeX Figure | LaTeX Line | Notebook Cell | Content |
|--------------|------------|---------------|---------|
| Figure 1 | ~175 | Cell 38 (left plot) | Predicted vs True scatter |
| Figure 2 | ~187 | Cell 38 (right plot) | Residual plot |
| Figure 3 | ~199 | Cell 28 | Training curves |
| Figure 4 | ~228 | Cell 32 | Gender analysis |
| Figure 5 | ~252 | Cell 34 | Confusion matrix |
| Figure 6 | ~264 | Cell 36 | Grad-CAM heatmaps |
| Figure 7 | ~289 | Cell 40 | Sample predictions |

---

## 🎨 Screenshot Tips

### Resolution & Quality:
- Use **high resolution** (at least 1920×1080 screen)
- Save as **PNG** format (better quality than JPG)
- Make sure text is **readable** in plots

### Cropping:
- Crop tightly around each plot
- Include axis labels, titles, and legends
- Remove unnecessary whitespace

### Including in LaTeX:
1. Save screenshots with descriptive names:
   - `fig1_predictions_vs_true.png`
   - `fig2_residual_plot.png`
   - `fig3_training_history.png`
   - `fig4_gender_analysis.png`
   - `fig5_confusion_matrix.png`
   - `fig6_gradcam.png`
   - `fig7_difficult_samples.png`

2. Replace the placeholder code in LaTeX:
```latex
% FROM THIS:
\fbox{\parbox{0.8\textwidth}{\centering
\vspace{3cm}
\textcolor{blue}{\textbf{[INSERT SCREENSHOT HERE]}}\\
...
\vspace{3cm}
}}

% TO THIS:
\includegraphics[width=0.8\textwidth]{fig1_predictions_vs_true.png}
```

---

## ✅ Checklist Before Compiling LaTeX

- [ ] All 7 screenshots captured from correct notebook cells
- [ ] Screenshots saved as PNG files
- [ ] Files named descriptively (fig1, fig2, etc.)
- [ ] Screenshots placed in same folder as `.tex` file
- [ ] Placeholder `\fbox` code replaced with `\includegraphics`
- [ ] Table 4 "Variable" entries updated with actual gender metrics from Cell 32 output
- [ ] Team names updated in author section
- [ ] Compiled successfully with pdflatex

---

## 📝 Additional Notes

### Gender Metrics (Table 4, LaTeX Line ~218):
From **Cell 32** text output, copy the actual values:
- Male MAE: ___ months
- Female MAE: ___ months  
- Male RMSE: ___ months
- Female RMSE: ___ months
- Male R²: ___
- Female R²: ___

Replace "Variable" entries in the LaTeX table with these values.

### Training Summary Data (Optional):
From **Cell 28** text output at the bottom, you can also note:
- Best epoch number
- Best validation loss
- Best validation MAE

This can be mentioned in the text near Figure 3.

---

## 🚀 Quick Workflow

1. **Open your Jupyter notebook**
2. **Run cells 28, 32, 34, 36, 38, 40** (or just run all cells)
3. **Take 7 screenshots** following the mapping above
4. **Save them** with clear names in your project folder
5. **Update LaTeX file** replacing placeholders with `\includegraphics`
6. **Compile** the report: `pdflatex Bone_Age_Report.tex`
7. **Done!** 🎉

---

Good luck with your report! 📄✨
