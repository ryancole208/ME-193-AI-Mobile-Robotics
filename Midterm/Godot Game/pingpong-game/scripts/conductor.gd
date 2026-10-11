class_name Conductor
extends AudioStreamPlayer
signal beat(beat_number: int, beat_in_bar: int)
var downbeat: int = 0

var offset: float = 0.0
@export_file("*.json") var chart_path: String
var chart: Dictionary = {}

var last_beat: int = -1
var song_position: float = 0.0
var output_latency: float = 0.0
var tempo_map: Array[Dictionary] = []
@export var debug_start_bar: int = 0

func _ready() -> void:
	output_latency = AudioServer.get_output_latency()
	chart = load_chart(chart_path)
	offset = chart["offset_ms"] / 1000.0
	downbeat = int(chart["downbeat"])
	build_tempo_map(chart["tempo"])
	var start_time: float = 0.0
	if debug_start_bar > 0:
		start_time = beat_to_seconds(bar_to_beat(debug_start_bar))
	play(start_time)


func _process(_delta: float) -> void:
	if playing:
		song_position = get_playback_position() \
			+ AudioServer.get_time_since_last_mix() - output_latency
		var current_beat: int = floori(seconds_to_beat(song_position))
		if current_beat > last_beat:
			last_beat = current_beat
			beat.emit(current_beat, posmod(current_beat - downbeat, 4))

func load_chart(path: String) -> Dictionary:
	var text: String = FileAccess.get_file_as_string(path)
	var data: Variant = JSON.parse_string(text)
	if data == null:
		push_error("Could not read chart: " + path)
		return {}
	return data
	
func build_tempo_map(entries: Array) -> void:
	tempo_map.clear()
	var time: float = offset
	for i in range(entries.size()):
		var entry: Dictionary = entries[i]
		if i > 0:
			var prev: Dictionary = tempo_map[i - 1]
			time += (entry["beat"] - prev["beat"]) * 60.0 / prev["bpm"]
		time += entry.get("shift_ms", 0.0) / 1000.0
		tempo_map.append({"beat": float(entry["beat"]), "bpm": float(entry["bpm"]), "time": time})

func seconds_to_beat(t: float) -> float:
	var seg: Dictionary = tempo_map[0]
	for s in tempo_map:
		if t >= s["time"]:
			seg = s
		else:
			break
	return seg["beat"] + (t - seg["time"]) * seg["bpm"] / 60.0

func beat_to_seconds(b: float) -> float:
	var seg: Dictionary = tempo_map[0]
	for s in tempo_map:
		if b >= s["beat"]:
			seg = s
		else:
			break
	return seg["time"] + (b - seg["beat"]) * 60.0 / seg["bpm"]

func bar_to_beat(bar: int) -> float:
	return downbeat + (bar - 1) * 4.0
