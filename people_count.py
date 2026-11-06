def get_current_people_count(result):

    if result.boxes is None:
        return 0

    if result.boxes.id is None:
        return 0

    track_ids = result.boxes.id.int().cpu().tolist()

    unique_ids = set(track_ids)

    return len(unique_ids)


def get_tracking_ids(result):

    if result.boxes is None:
        return []

    if result.boxes.id is None:
        return []

    return result.boxes.id.int().cpu().tolist()