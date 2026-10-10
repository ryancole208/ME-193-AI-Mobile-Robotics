extends AudioStreamPlayer
signal beat(beat_number: int)

@export var bpm: float = 120.0
@export var offset: float = 0.0
var sec_per_beat: float = 0.5
var last_beat: int = -1
var song_position: float = 0.0
var output_latency: float = 0.0

func _ready() -> void:
	output_latency = AudioServer.get_output_latency()
	sec_per_beat = 60.0 / bpm

func _process(_delta: float) -> void:
	if playing:
		song_position = get_playback_position() \
			+ AudioServer.get_time_since_last_mix() - output_latency
		var current_beat: int = floori((song_position - offset) / sec_per_beat)
		if current_beat > last_beat:
			last_beat = current_beat
			beat.emit(current_beat)
