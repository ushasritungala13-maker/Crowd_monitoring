def classify_zone(density):

    if density == "LOW":
        return "GREEN"

    elif density == "MEDIUM":
        return "YELLOW"

    else:
        return "RED"


def get_alert(zone):

    if zone == "GREEN":
        return "NORMAL"

    elif zone == "YELLOW":
        return "MONITOR CROWD"

    else:
        return "HIGH CROWD ALERT"