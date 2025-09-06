
"""
Assignment 1a: Student Marks Analysis (Excel Automation)
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import openpyxl 
import xlsxwriter

INPUT_XLSX = Path(r"E:/Vectra-Assignment/students-data/student.xlsx")
OUTPUT_XLSX = Path(r"E:/Vectra-Assignment/students-data/results.xlsx")

def create_input_excel_if_missing(path: Path):
    
    if path.exists():
        print(f" We Found existing input Excel at: {path}")
        return

    print(" The Input Excel not found. Creating sample 'student.xlsx' from provided dictionary...")
    data = {
        "StudentID": [101, 102, 103, 104, 105, 106, 107],
        "Name": ["Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace"],
        "Math": [95, 72, 88, 55, 80, 99, 68],
        "Physics": [89, 65, 91, 62, 77, 95, 72],
        "Chemistry": [92, 70, 85, 58, 79, 97, 74],
        "Biology": [88, 60, 90, 61, 83, 96, 70],
    }
    df = pd.DataFrame(data)
    df.to_excel(path, sheet_name="My_Sheet1", index=False)
    print(f" Now Created the input Excel at: {path}\n")


def compute_scores_vectorized(df: pd.DataFrame, subject_cols):
    """
    Now the Task is to Compute Total, Average, and Grade using NumPy vectorized operations.
    Which Returns a new DataFrame.
    """
    print(" Now Performing vectorized computations for Total, Average and Grade...")
    # Convert subject columns to a NumPy array (shape: n_students x n_subjects)
    scores = df[subject_cols].to_numpy(dtype=float) 

    # Total score per student (vectorized sum across axis=1)
    totals = scores.sum(axis=1)  

    # Average score per student (vectorized mean across axis=1)
    averages = scores.mean(axis=1) 

    # Assign grades using numpy.select (vectorized conditional assignment)
    conditions = [
        averages >= 90,
        (averages >= 75) & (averages < 90),
        (averages >= 60) & (averages < 75),
        averages < 60,
    ]
    choices = ["A", "B", "C", "F"]
    grades = np.select(conditions, choices, default="F") 

    # Attach results to a copy of DataFrame and return
    df_out = df.copy()
    df_out["Total"] = totals
    # Round average to 2 decimals 
    df_out["Average"] = np.round(averages, 2)
    df_out["Grade"] = grades

    print(" The Vectorized computations were completed .\n")
    return df_out


def find_top_performers(df: pd.DataFrame, subject_cols, top_k=3):
    """
    Now the Task is to Identify top 3 students per subject.
    """
    print(f"Then Finding top {top_k} performers per subject...")
    rows = []
    for subject in subject_cols:
        
        top_df = df.sort_values(by=subject, ascending=False).head(top_k)
        for rank, (_, r) in enumerate(top_df.iterrows(), start=1):
            rows.append({
                "Subject": subject,
                "Rank": rank,
                "StudentID": int(r["StudentID"]),
                "Name": r["Name"],
                "Score": r[subject],
            })
    top_performers_df = pd.DataFrame(rows)
    print("The Top performers computed.\n")
    return top_performers_df


def save_results_to_excel(summary_df: pd.DataFrame, top_perf_df: pd.DataFrame, averages_series, output_path: Path):
    """
    Now the Task is to Write Summary and Top Performers to a single Excel workbook.
    and Also to create bar chart (Average marks per subject) embedded in the Top Performers sheet.
    """
    print(f" Here we are writing results to Excel workbook: {output_path} ...")
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
    
        summary_to_write = summary_df[["StudentID", "Name", "Total", "Average", "Grade"]]
        summary_to_write.to_excel(writer, sheet_name="Summary", index=False)
        print(" The Summary sheet is done.")


        top_startrow = 0
        top_perf_df.to_excel(writer, sheet_name="Top Performers", index=False, startrow=top_startrow)
        rows_used_by_top = top_perf_df.shape[0] + 1  # +1 for header row

        
        avg_df = pd.DataFrame({
            "Subject": averages_series.index,
            "Average": np.round(averages_series.values, 2)
        })

        avg_startrow = rows_used_by_top + 2  # leave a blank row after top performers table
        avg_df.to_excel(writer, sheet_name="Top Performers", index=False, startrow=avg_startrow)
        print("Now The Top Performers sheet is done.")

        # Creating a bar chart for average marks per subject
        workbook  = writer.book
        worksheet = writer.sheets["Top Performers"]

        
        chart = workbook.add_chart({"type": "column"})
        n_subjects = len(avg_df)
    
        chart.add_series({
            "name": "Average Marks per Subject",
            "categories": ["Top Performers", avg_startrow + 1, 0, avg_startrow + n_subjects, 0],
            "values":     ["Top Performers", avg_startrow + 1, 1, avg_startrow + n_subjects, 1],
            "gap": 20,
        })
        chart.set_title({"name": "Average Marks Per Subject (All Students)"})
        chart.set_x_axis({"name": "Subject"})
        chart.set_y_axis({"name": "Average Score", "major_gridlines": {"visible": False}})
        chart.set_legend({"position": "none"})

        # Inserting chart to the sheet 
        chart_row = avg_startrow
        chart_col = 4  # column E (0-indexed)
        worksheet.insert_chart(chart_row, chart_col, chart, {"x_scale": 1.2, "y_scale": 1.2})
        print(" Finally The Embedded bar chart created in the 'Top Performers' sheet.")


    print(f"Finally All Results saved were to: {output_path}\n")


def main():
    print("\n=== Vectra Assignment 1a: Student Marks Analysis ===\n")
    
    create_input_excel_if_missing(INPUT_XLSX)

    # 1) Loading data from Excel
    print(" Loading data from Excel input (student.xlsx)...")
    df = pd.read_excel(INPUT_XLSX)
    print("The Data was loaded successfully. Here are the first few rows: ")
    print(df.head().to_string(index=False))
    print("")

    
    subject_cols = ["Math", "Physics", "Chemistry", "Biology"]

    # 2) Vectorized computations for Total, Average, Grade
    processed_df = compute_scores_vectorized(df, subject_cols)

    # Printing summary table (StudentID, Name, Total, Average, Grade)
    print("The Result Summary (StudentID, Name, Total, Average, Grade):")
    print(processed_df[["StudentID", "Name", "Total", "Average", "Grade"]].to_string(index=False))
    print("")


    # 3) Finding Top Performers (per subject)
    top_performers_df = find_top_performers(df, subject_cols, top_k=3)
    print("The Top performers (tidy table):")
    print(top_performers_df.to_string(index=False))
    print("")


    # 4) Then Data Visualization — computing average marks per subject 
    avg_per_subject = df[subject_cols].mean()
    print("Then Average marks per subject :")
    for sub, avg in avg_per_subject.items():
        print(f"   - {sub}: {avg:.2f}")
    print("")


    # 5) Finally, Saving Results back to Excel (two sheets + embedded chart)
    save_results_to_excel(processed_df, top_performers_df, avg_per_subject, OUTPUT_XLSX)


if __name__ == "__main__":
    main()
