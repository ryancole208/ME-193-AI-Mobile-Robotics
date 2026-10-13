extends Control

@export var conductor: Conductor
@export var taps_needed: int = 16
var offsets: Array[float] = []
var result_ms: float = 0.0

func _ready() -> void:
	Bridge.swing_received.connect(_on_bridge_swing)

func _on_bridge_swing(age_s: float, _strength: float) -> void:
	record_tap(age_s)

func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_cancel"):
		Fader.go("res://scenes/Title.tscn")
	elif event.is_action_pressed("swing"):
		record_tap(0.0)
	elif event.is_action_pressed("ui_accept") and offsets.size() >= taps_needed:
		GameState.save_input_offset(result_ms)
		Fader.go("res://scenes/Title.tscn")

func record_tap(age_s: float) -> void:
	if conductor.counting_in or offsets.size() >= taps_needed:
		return
	var pos: float = conductor.song_position - age_s - GameState.input_offset_ms / 1000.0
	var beat: float = conductor.seconds_to_beat(pos)
	var nearest: float = roundf(beat)
	var offset_ms: float = (pos - conductor.beat_to_seconds(nearest)) * 1000.0
	offsets.append(offset_ms)
	$Timeline.add_offset(offset_ms)
	$Info.text = "TAP %d / %d\nLAST %+d ms" % [offsets.size(), taps_needed, roundi(offset_ms)]
	if offsets.size() >= taps_needed:
		finish()

func finish() -> void:
	var sorted: Array[float] = offsets.duplicate()
	sorted.sort()
	result_ms = GameState.input_offset_ms + sorted[int(sorted.size() / 2.0)]
	$Info.text = "OFFSET %+d ms (CHANGE %+d)\nENTER: SAVE   ESC: CANCEL" % [
		roundi(result_ms), roundi(result_ms - GameState.input_offset_ms)]
	conductor.stop()
