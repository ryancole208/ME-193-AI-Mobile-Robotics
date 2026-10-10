extends AudioStreamPlayer
signal beat(beat_number: int, beat_in_bar: int)
var downbeat: int = 0
var bpm: float = 120.0
var offset: float = 0.0
@export_file("*.json") var chart_path: String
var chart: Dictionary = {}
var sec_per_beat: float = 0.5
var last_beat: int = -1
var song_position: float = 0.0
var output_latency: float = 0.0

func _ready() -> void:
	output_latency = AudioServer.get_output_latency()
	chart = load_chart(chart_path)
	offset = chart["offset_ms"] / 1000.0
	downbeat = int(chart["downbeat"])
	bpm = chart["tempo"][0]["bpm"]
	sec_per_beat = 60.0 / bpm


func _process(_delta: float) -> void:
	if playing:
		song_position = get_playback_position() \
			+ AudioServer.get_time_since_last_mix() - output_latency
		var current_beat: int = floori((song_position - offset) / sec_per_beat)
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
