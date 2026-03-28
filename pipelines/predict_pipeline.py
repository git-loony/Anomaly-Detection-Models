def predict_single(model_ae, model_if, scaler, x):
    x_scaled = scaler.transform([x])

    ae_score = get_ae_scores(model_ae, x_scaled)[0]
    if_score = model_if.score(x_scaled)[0]

    final = 0.6 * ae_score + 0.4 * if_score
    return final