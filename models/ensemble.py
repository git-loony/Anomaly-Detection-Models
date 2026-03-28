def weighted_score(ae_score, if_score, rule_score=None):
    w_ae = 0.6
    w_if = 0.4

    final = w_ae * ae_score + w_if * if_score

    if rule_score is not None:
        final += 0.2 * rule_score

    return final