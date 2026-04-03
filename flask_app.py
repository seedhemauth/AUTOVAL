from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route("/api/options")
def get_options():
    import pandas as pd
    df = pd.read_csv("cardekho_dataset.csv")
    df.columns = df.columns.str.strip().str.lower()
    brands = sorted(df["brand"].dropna().unique())

    models = {}
    for brand in brands:
        models[brand] = sorted(df[df["brand"] == brand]["model"].dropna().unique())
    return jsonify({
        "brands": brands,
        "models": models
    })

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        input_dict = {
            "age": data["age"],
            "mileage": data["mileage"],
            "engine": data["engine"],
            "max_power": data["max_power"],
            "seats": data["seats"],
            "owner": data.get("owner")
        }

        # Add categorical
        input_dict[f"fuel_{data['fuel']}"] = 1
        input_dict[f"transmission_{data['transmission']}"] = 1
        input_dict[f"brand_{data['brand']}"] = 1
        input_dict[f"model_{data['model']}"] = 1

        owner_val = data.get("owner")
        input_dict[f"owner_{owner_val}"] = 1

        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=columns, fill_value=0)

        prediction_log = model.predict(input_df)[0]
        prediction = np.expm1(prediction_log)

        low = prediction * 0.9
        high = prediction * 1.1

        return jsonify({
            "success": True,
            "exact": f"₹ {prediction:,.0f}",
            "range": f"₹ {low:,.0f} - ₹ {high:,.0f}",
            "display": f"{round(prediction/100000, 1)} Lakh"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)