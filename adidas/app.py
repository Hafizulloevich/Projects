import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import datetime


df = pd.read_excel("Adidas.xlsx")
df["Month_Year"] = df["InvoiceDate"].dt.strftime("%b'%y")


def retailer_sales():
    fig = px.bar(df, x="Retailer", y="TotalSales",
                 labels={"TotalSales": "Total Sales ($)"},
                 title="Total Sales by Retailer",
                 template="gridon")
    data = df.groupby("Retailer")["TotalSales"].sum().reset_index()
    return fig, data

def monthly_sales():
    result = df.groupby("Month_Year")["TotalSales"].sum().reset_index()
    fig = px.line(result, x="Month_Year", y="TotalSales",
                  title="Total Sales Over Time", template="gridon")
    return fig, result

def sales_by_state():
    result1 = df.groupby("State")[["TotalSales","UnitsSold"]].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=result1["State"], y=result1["TotalSales"], name="Total Sales"))
    fig.add_trace(go.Scatter(x=result1["State"], y=result1["UnitsSold"], mode="lines",
                             name="Units Sold", yaxis="y2"))
    fig.update_layout(
        title="Total Sales and Units Sold by State",
        yaxis=dict(title="Total Sales"),
        yaxis2=dict(title="Units Sold", overlaying="y", side="right"),
        template="gridon"
    )
    return fig, result1

def region_city_sales():
    treemap = df.groupby(["Region","City"])["TotalSales"].sum().reset_index()
    fig = px.treemap(treemap, path=["Region","City"], values="TotalSales",
                     title="Total Sales by Region and City")
    return fig, treemap


with gr.Blocks() as demo:
    gr.Markdown("# 👟 Adidas Interactive Sales Dashboard")
    gr.Markdown(f"📅 Last updated: **{datetime.datetime.now().strftime('%d %B %Y')}**")

    with gr.Tab("Retailer Sales"):
        plot, table = gr.Plot(), gr.DataFrame()
        btn = gr.Button("Show Retailer Sales")
        btn.click(fn=retailer_sales, outputs=[plot, table])

    with gr.Tab("Monthly Sales"):
        plot2, table2 = gr.Plot(), gr.DataFrame()
        btn2 = gr.Button("Show Monthly Sales")
        btn2.click(fn=monthly_sales, outputs=[plot2, table2])

    with gr.Tab("Sales by State"):
        plot3, table3 = gr.Plot(), gr.DataFrame()
        btn3 = gr.Button("Show State Sales")
        btn3.click(fn=sales_by_state, outputs=[plot3, table3])

    with gr.Tab("Region & City Sales"):
        plot4, table4 = gr.Plot(), gr.DataFrame()
        btn4 = gr.Button("Show Region/City Sales")
        btn4.click(fn=region_city_sales, outputs=[plot4, table4])

demo.launch()
