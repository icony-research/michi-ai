import supervision as sv


def get_triggering_anchors(config):
    """設定からアンカーポイントを取得"""
    anchor_config = config['tracking'].get('triggering_anchors', 'BOTTOM_CENTER')

    anchor_map = {
        'CENTER': [sv.Position.CENTER],
        'BOTTOM_CENTER': [sv.Position.BOTTOM_CENTER],
        'TOP_CENTER': [sv.Position.TOP_CENTER],
        'CENTER_OF_MASS': [sv.Position.CENTER_OF_MASS],
        'BOTTOM_LEFT': [sv.Position.BOTTOM_LEFT],
        'BOTTOM_RIGHT': [sv.Position.BOTTOM_RIGHT],
        'BOTTOM_CORNERS': [sv.Position.BOTTOM_LEFT, sv.Position.BOTTOM_RIGHT],
        'ALL_CORNERS': [sv.Position.TOP_LEFT, sv.Position.TOP_RIGHT,
                        sv.Position.BOTTOM_LEFT, sv.Position.BOTTOM_RIGHT],
        'DEFAULT': [sv.Position.TOP_LEFT, sv.Position.TOP_RIGHT,
                    sv.Position.BOTTOM_LEFT, sv.Position.BOTTOM_RIGHT],
    }

    return anchor_map.get(anchor_config, anchor_map['BOTTOM_CENTER'])


def setup_detection_components(config, fps, log_func=None):
    """検出・追跡・可視化コンポーネントの設定"""
    log = log_func if callable(log_func) else (lambda *_, **__: None)

    triggering_anchors = get_triggering_anchors(config)

    anchor_names = [str(anchor).split('.')[-1] for anchor in triggering_anchors]
    log("✓ 通過判定設定:")
    log(f"  - アンカーポイント: {', '.join(anchor_names)}")

    if 'lines' in config:
        line_mode = config['lines'].get('mode', 'dual')

        if line_mode == 'single':
            up_start = sv.Point(config['lines']['up_line']['start_x'], config['lines']['up_line']['start_y'])
            up_end = sv.Point(config['lines']['up_line']['end_x'], config['lines']['up_line']['end_y'])
            line_zone = sv.LineZone(start=up_start, end=up_end, triggering_anchors=triggering_anchors)
            line_annotator = sv.LineZoneAnnotator(thickness=config['lines']['up_line']['thickness'], color=sv.Color.GREEN)

            line_zones = {'single': line_zone}
            line_annotators = {'single': line_annotator}
        else:
            up_start = sv.Point(config['lines']['up_line']['start_x'], config['lines']['up_line']['start_y'])
            up_end = sv.Point(config['lines']['up_line']['end_x'], config['lines']['up_line']['end_y'])
            up_line_zone = sv.LineZone(start=up_start, end=up_end, triggering_anchors=triggering_anchors)
            up_line_annotator = sv.LineZoneAnnotator(thickness=config['lines']['up_line']['thickness'], color=sv.Color.GREEN)

            down_start = sv.Point(config['lines']['down_line']['start_x'], config['lines']['down_line']['start_y'])
            down_end = sv.Point(config['lines']['down_line']['end_x'], config['lines']['down_line']['end_y'])
            down_line_zone = sv.LineZone(start=down_start, end=down_end, triggering_anchors=triggering_anchors)
            down_line_annotator = sv.LineZoneAnnotator(thickness=config['lines']['down_line']['thickness'], color=sv.Color.BLUE)

            line_zones = {'up': up_line_zone, 'down': down_line_zone}
            line_annotators = {'up': up_line_annotator, 'down': down_line_annotator}
    else:
        line_start = sv.Point(config['line']['start_x'], config['line']['start_y'])
        line_end = sv.Point(config['line']['end_x'], config['line']['end_y'])
        line_zone = sv.LineZone(start=line_start, end=line_end, triggering_anchors=triggering_anchors)
        line_annotator = sv.LineZoneAnnotator(thickness=config['line']['thickness'])

        line_zones = {'single': line_zone}
        line_annotators = {'single': line_annotator}

    tracker = sv.ByteTrack(
        track_activation_threshold=config['tracking']['track_activation_threshold'],
        lost_track_buffer=config['tracking']['lost_track_buffer'],
        minimum_matching_threshold=config['tracking']['minimum_matching_threshold'],
        frame_rate=int(fps)
    )

    box_annotator = sv.BoxAnnotator(thickness=config['visualization']['box_thickness'])
    label_annotator = sv.LabelAnnotator(
        text_scale=config['visualization']['label_text_scale'],
        text_thickness=config['visualization']['label_text_thickness'],
        text_padding=config['visualization']['label_text_padding']
    )

    return line_zones, line_annotators, tracker, box_annotator, label_annotator
