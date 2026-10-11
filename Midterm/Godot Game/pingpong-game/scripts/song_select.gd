extends Control
const GREY := Color("C2C3C7")
const YELLOW := Color("FFEC27")

func _ready() -> void:
	$List.add_child(make_columns("", ["BPM", "TIME", "CUES", "BEST"]))
	for file in DirAccess.get_files_at("res://charts"):
		if not file.ends_with(".json"):
			continue
		var path: String = "res://charts/" + file
		var chart: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(path))
		var button: Button = make_row(chart)
		button.pressed.connect(_on_song_chosen.bind(path))
		$List.add_child(button)
	$List.get_child(1).grab_focus()

func _on_song_chosen(path: String) -> void:
	GameState.chart_path = path
	get_tree().change_scene_to_file("res://scenes/main.tscn")
func make_row(chart: Dictionary) -> Button:
	var button := Button.new()
	button.custom_minimum_size = Vector2(372, 14)
	var author: String = chart["author"].get_slice(" (", 0)
	var row := make_columns(chart["title"] + "  " + author,
		[bpm_text(chart).trim_suffix(" BPM"), length_text(chart), str(cue_count(chart)), best_text(chart)])
	row.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	row.mouse_filter = Control.MOUSE_FILTER_IGNORE
	button.add_child(row)
	row.modulate = GREY
	button.focus_entered.connect(func(): row.modulate = YELLOW)
	button.focus_exited.connect(func(): row.modulate = GREY)
	return button

func make_label(text: String) -> Label:
	var label := Label.new()
	label.text = text
	label.add_theme_font_size_override("font_size", 8)
	return label

func bpm_text(chart: Dictionary) -> String:
	var bpms: Array = chart["tempo"].map(func(e): return roundi(e["bpm"]))
	var low: int = bpms.min()
	var high: int = bpms.max()
	return "%d BPM" % low if low == high else "%d-%d BPM" % [low, high]
	
func length_text(chart: Dictionary) -> String:
	var song: AudioStream = load("res://audio/music/" + chart["song"])
	var secs: int = roundi(song.get_length())
	return "%d:%02d" % [floori(secs / 60.0), secs % 60]

func cue_count(chart: Dictionary) -> int:
	var count: int = 0
	for bar in range(int(chart["start_bar"]), int(chart["end_bar"]) + 1):
		count += ChartTools.pattern_for_bar(chart, bar).size()
	return count

const COL_WIDTH := 28

func make_columns(left_text: String, cols: Array) -> HBoxContainer:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 4)
	var left: Label = make_label(left_text)
	left.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	left.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
	row.add_child(left)
	for text in cols:
		var col: Label = make_label(text)
		col.custom_minimum_size.x = COL_WIDTH
		col.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
		row.add_child(col)
	return row

func best_text(chart: Dictionary) -> String:
	var hs: float = GameState.high_score(chart["title"])
	return "--" if hs < 0.0 else "%d%%" % roundi(hs)
