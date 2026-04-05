import gradio as gr
import pandas as pd
from predict import predict
import matplotlib.pyplot as plt

def run(file):
    df = predict(file.name)

    # Summary
    total = len(df)
    anomalies = int(df["is_anomaly"].sum())
    percent = round(anomalies / total * 100, 2)

    summary = {
        "Total Rows": total,
        "Anomalies": anomalies,
        "Anomaly %": percent
    }

    # Plot
    plt.figure()
    df["final_score"].hist()
    plt.title("Anomaly Score Distribution")
    plot_path = "plot.png"
    plt.savefig(plot_path)
    plt.close()

    return df.head(50), summary, plot_path

demo = gr.Interface(
    fn=run,
    inputs=gr.File(file_types=[".csv"]),
    outputs=[
        gr.Dataframe(label="Preview"),
        gr.JSON(label="Summary"),
        gr.Image(label="Score Distribution")
    ],
    title="🚨 Anomaly Detection Dashboard"
)

if __name__ == "__main__":
    demo.launch()