import PySimpleGUI as sg
from sentence_transformers import SentenceTransformer, util
import numpy as np
import threading

MODEL_OPTIONS = [
    "all-MiniLM-L6-v2",
    "paraphrase-MiniLM-L6-v2"
]

def compute_similarity(model_name, sentences, window):
    try:
        window.write_event_value("-LOG-", f"Загружаем модель {model_name} ...")
        model = SentenceTransformer(model_name)
        window.write_event_value("-LOG-", "Вычисляем эмбеддинги ...")
        embeddings = model.encode(sentences, convert_to_tensor=True)
        window.write_event_value("-LOG-", "Вычисляем косинусную матрицу ...")
        cos_scores = util.cos_sim(embeddings, embeddings).cpu().numpy()
        # Prepare similarity text
        n = len(sentences)
        header = ["#"] + [f"{i+1}" for i in range(n)]
        rows = [header]
        for i in range(n):
            row = [str(i+1)] + [f"{cos_scores[i][j]:.4f}" for j in range(n)]
            rows.append(row)
        # Top pairs (excluding diagonal)
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                pairs.append((i, j, float(cos_scores[i][j])))
        pairs.sort(key=lambda x: x[2], reverse=True)
        top_lines = ["Top similar pairs (index1, index2, score):"]
        for a,b,sc in pairs[:10]:
            top_lines.append(f"{a+1}\t{b+1}\t{sc:.4f}\t| {sentences[a][:60]} ... | {sentences[b][:60]} ...")
        result_text = "Similarity matrix (rows = sentences):\n\n"
        for r in rows:
            result_text += "\t".join(r) + "\n"
        result_text += "\n" + "\n".join(top_lines)
        window.write_event_value("-RESULT-", result_text)
        window.write_event_value("-LOG-", "Готово.")
    except Exception as e:
        window.write_event_value("-ERROR-", str(e))

def start_background(model_name, sentences, window):
    thread = threading.Thread(target=compute_similarity, args=(model_name, sentences, window), daemon=True)
    thread.start()

def build_layout():
    sg.theme("DarkAmber")
    layout = [
        [sg.Text("Введите предложения (одно предложение на строку):")],
        [sg.Multiline(key="-INPUT-", size=(80,15), tooltip="One sentence per line")],
        [sg.Text("Выберите модель:"), sg.Combo(MODEL_OPTIONS, default_value=MODEL_OPTIONS[0], key="-MODEL-"), sg.Button("Анализ", key="-ANALYZE-")],
        [sg.Frame("Лог", [[sg.Multiline(key="-LOGBOX-", size=(80,5), disabled=True)]])],
        [sg.Frame("Результат", [[sg.Multiline(key="-OUTPUT-", size=(80,15), disabled=True)]])],
        [sg.Button("Выйти")]
    ]
    return layout

def main():
    layout = build_layout()
    window = sg.Window("Semantic Analyzer (sentence-transformers)", layout, finalize=True)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == "Выйти":
            break
        if event == "-ANALYZE-":
            raw = values["-INPUT-"].strip()
            if not raw:
                sg.popup("Введите хотя бы одно предложение (одна строка — одно предложение).")
                continue
            sentences = [s.strip() for s in raw.splitlines() if s.strip()]
            if len(sentences) < 1:
                sg.popup("Введите хотя бы одно предложение.")
                continue
            model_name = values["-MODEL-"] or MODEL_OPTIONS[0]
            window["-LOGBOX-"].update("Запуск анализа...\n")
            start_background(model_name, sentences, window)
        if event == "-LOG-":
            prev = window["-LOGBOX-"].get()
            new = prev + values["-LOG-"] + "\n"
            window["-LOGBOX-"].update(new)
        if event == "-RESULT-":
            window["-OUTPUT-"].update(values["-RESULT-"])
        if event == "-ERROR-":
            sg.popup_error("Ошибка при выполнении:", values["-ERROR-"])

    window.close()

if __name__ == "__main__":
    main()
