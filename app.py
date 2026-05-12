import gradio as gr
import pandas as pd
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

def predict(age, job, marital, education, default, housing, loan,
            contact, month, day_of_week, duration, campaign,
            pdays, previous, poutcome, emp_var_rate, cons_price_idx,
            cons_conf_idx, euribor3m, nr_employed):

    input_dict = {col: 0 for col in columns}

    # Sayısal değerler
    input_dict["age"] = age
    input_dict["duration"] = duration
    input_dict["campaign"] = campaign
    input_dict["pdays"] = pdays
    input_dict["previous"] = previous
    input_dict["emp.var.rate"] = emp_var_rate
    input_dict["cons.price.idx"] = cons_price_idx
    input_dict["cons.conf.idx"] = cons_conf_idx
    input_dict["euribor3m"] = euribor3m
    input_dict["nr.employed"] = nr_employed

    # Türetilen özellikler
    input_dict["was_contacted_before"] = int(pdays != 999)
    input_dict["contact_intensity"] = campaign / (previous + 1)
    input_dict["economic_pressure"] = euribor3m * emp_var_rate
    if duration <= 100:
        input_dict["duration_cat"] = 0
    elif duration <= 300:
        input_dict["duration_cat"] = 1
    elif duration <= 600:
        input_dict["duration_cat"] = 2
    else:
        input_dict["duration_cat"] = 3

    # One-hot sütunlar
    for key in [f"job_{job}", f"marital_{marital}", f"education_{education}",
                f"default_{default}", f"housing_{housing}", f"loan_{loan}",
                f"contact_{contact}", f"month_{month}", f"day_of_week_{day_of_week}",
                f"poutcome_{poutcome}"]:
        if key in input_dict:
            input_dict[key] = 1

    df_input = pd.DataFrame([input_dict])
    df_input = df_input.reindex(
        columns=columns,
        fill_value=0
    )
    prob = model.predict_proba(df_input)[0][1]

    return {
        "✅ Abone Olur": round(float(prob), 3),
        "❌ Olmaz": round(float(1 - prob), 3)
    }

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(18, 95, value=40, label="Yaş"),
        gr.Dropdown(["admin.", "blue-collar", "technician", "retired",
                     "management", "entrepreneur", "self-employed",
                     "housemaid", "services", "student", "unemployed"], value="admin.", label="Meslek"),
        gr.Dropdown(["married", "single", "divorced"], value="married", label="Medeni Hal"),
        gr.Dropdown(["university.degree", "high.school", "basic.9y",
                     "basic.6y", "basic.4y", "professional.course",
                     "illiterate"], value="university.degree", label="Eğitim"),
        gr.Dropdown(["no", "yes", "unknown"], value="no", label="Temerrüt Kredisi"),
        gr.Dropdown(["yes", "no", "unknown"], value="yes", label="Konut Kredisi"),
        gr.Dropdown(["no", "yes", "unknown"], value="no", label="Kişisel Kredi"),
        gr.Dropdown(["cellular", "telephone"], value="cellular", label="İletişim Türü"),
        gr.Dropdown(["jan","feb","mar","apr","may","jun",
                     "jul","aug","sep","oct","nov","dec"], value="may", label="Ay"),
        gr.Dropdown(["mon","tue","wed","thu","fri"], value="mon", label="Haftanın Günü"),
        gr.Slider(0, 5000, value=200, label="Arama Süresi (sn)"),
        gr.Slider(1, 50, value=2, label="Kampanya Temas Sayısı"),
        gr.Slider(0, 999, value=999, label="Önceki Kampanyadan Geçen Gün (999=hiç)"),
        gr.Slider(0, 10, value=0, label="Önceki Kampanya Temas Sayısı"),
        gr.Dropdown(["nonexistent", "failure", "success"], value="nonexistent", label="Önceki Kampanya Sonucu"),
        gr.Slider(-3.5, 1.5, value=1.1, step=0.1, label="İstihdam Değişim Oranı"),
        gr.Slider(92.0, 95.0, value=93.5, step=0.1, label="Tüketici Fiyat Endeksi"),
        gr.Slider(-51.0, -26.0, value=-36.0, step=0.1, label="Tüketici Güven Endeksi"),
        gr.Slider(0.6, 5.1, value=4.8, step=0.01, label="Euribor 3 Aylık"),
        gr.Slider(4900, 5300, value=5100, step=1, label="Çalışan Sayısı"),
    ],
    outputs=gr.Label(label="Tahmin Sonucu"),
    title="🏦 Banka Pazarlama Tahmin Modeli",
    description="Müşterinin vadeli mevduat teklifini kabul edip etmeyeceğini tahmin eder. (AUC-ROC: 0.9496)",
    theme=gr.themes.Soft()
)

iface.launch(
    server_name="0.0.0.0",
    server_port=7860
)