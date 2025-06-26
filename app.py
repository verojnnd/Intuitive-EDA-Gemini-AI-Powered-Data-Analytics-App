# pip install -r requirements.txt
# chainlit run main.py

import os, io, asyncio, tempfile, traceback
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import chainlit as cl
from PIL import Image
import google.generativeai as genai
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
GEMINI_MODEL = "gemini-1.5-flash-latest"
GEMINI_AVAILABLE = False

# Google AI Studio API Key
os.environ["GOOGLE_API_KEY"] = "" # Fill with your own API Key

try:
    if api_key := os.environ.get("GOOGLE_API_KEY"):
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        GEMINI_AVAILABLE = True
except Exception as e:
    print(f"Error configuring Gemini model: {e}")

def save_fig(fig):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(f.name, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return f.name

def df_info_string(df, max_rows=5):
    buf = io.StringIO()
    df.info(buf=buf)
    schema = buf.getvalue()
    head = df.head(max_rows).to_markdown(index=False)

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing_info = "No missing values" if missing.empty else str(missing)
    return f"### Schema:\n```\n{schema}```\n\n### Preview:\n{head}\n\n### Missing Values:\n{missing_info}"

async def ai_text_analysis(prompt_type, df_context):
    if not GEMINI_AVAILABLE: return f"Gemini AI not available."
    prompts = {
        "plan": f"You are a data analyst. Suggest a concise data analysis plan:\n{df_context}",
        "final": f"Summarize insights from the following dataset:\n{df_context}",
    }

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        res = await model.generate_content_async(
            prompts.get(prompt_type), generation_config=genai.types.GenerationConfig(max_output_tokens=500, temperature=0.3)
        )
        return res.text if res.parts else "Gemini response blocked."
    except Exception as e:
        # print(f"Error generating AI text: {e}")
        return f"Error generating AI text: {e}"
    
async def ai_vision_analysis(img_paths):
    if not GEMINI_AVAILABLE: return [("AI Vision", "Gemini not available.")]
    model = genai.GenerativeModel(GEMINI_MODEL)
    results = []                              
    
    for title, path in img_paths:
        try:
            img = Image.open(path)
            res = await model.generate_content_async([f"Explain this '{title}'", img],
                                                     generation_config=genai.types.GenerationConfig(max_output_tokens=200, temperature=0.2))
            results.append((title, res.text if res.parts else "Blocked or empty response."))
        
        except Exception as e:
            results.append((title, f"Error processing image: {e}"))
    return results

def generate_visuals(df):
    visualizations = []
    saved_files = []

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [col for col in df.select_dtypes('object') if 1 < df[col].nunique() < 30]
    
    try:
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap")
            path = save_fig(fig)
            visualizations.append(("Correlation Heatmap", path))
            saved_files.append(path)

        if len(numeric_cols) >= 3:
            sns.set(style="ticks")
            fig = sns.pairplot(df[numeric_cols[:5]].dropna()).fig
            fig.suptitle("Pairplot of Numeric Features", y=1.02)
            path = save_fig(fig)
            visualizations.append(("Pairplot", path))
            saved_files.append(path)

        for col in numeric_cols[:3]:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.violinplot(data=df, y=col, ax=ax)
            ax.set_title(f"Violin Plot for {col}")
            path = save_fig(fig)
            visualizations.append((f"Violin Plot - {col}", path))
            saved_files.append(path)

    except Exception as e:
        print(f"Visualization error: {e}")
        plt.close('all')

    return visualizations, saved_files

async def cleanup(files):
    for f in files:
        try: os.remove(f)
        except: pass

@cl.on_chat_start
async def start():
    await cl.Message(content="Upload a CSV file for AI analysis using Gemini.").send()
    files = await cl.AskFileMessage(content="Upload a CSV file", accept=["text/csv"]).send()

    if not files:
        return await cl.Message(content="No file received.").send()
    
    processing_msg = cl.Message(content="Processing...")
    await processing_msg.send()

    try:
        # content = files[0].content.decode("utf-8", errors="replace")
        # df = pd.read_csv(io.StringIO(content))
        df = pd.read_csv(files[0].path)
        if df.empty:
            # await processing_msg.update(content="Empty dataset.")
            await cl.Message(content="Empty dataset.").send()
            return
        
        cl.user_session.set("df", df)

        info = df_info_string(df)
        await cl.Message(content=info).send()

        if GEMINI_AVAILABLE:
            plan = await ai_text_analysis("plan", info)
            await cl.Message(content=f"### AI Plan:\n{plan}").send()
        
        visuals, saved_files = generate_visuals(df)
        for title, path in visuals:
            await cl.Message(content=f"**{title}**", elements=[cl.Image(name=title, path=path)]).send()

        if GEMINI_AVAILABLE:
            insights = await ai_vision_analysis(visuals)
            for title, insight in insights:
                await cl.Message(content=f"### {title} Insight\n{insight}").send()
            final = await ai_text_analysis("final", info)
            await cl.Message(content=f"### Final AI Report:\n{final}").send()

        # await processing_msg.update(content="Analysis complete.")
        await cl.Message(content=f"Analysis complete.").send()
        await cleanup([p for _, p in visuals])

    except Exception as e:
        traceback.print_exc()
        await cl.Message(content=f"Error: {e}").send()
