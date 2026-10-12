extends Control

@export var conductor: Conductor
@export var taps_needed: int = 16
var offsets: Array[float] = []
var result_ms: float = 0.0

func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_cancel"):
		get_tree().change_scene_to_file("res://scenes/Title.tscn")
	elif event.is_action_pressed("swing"):
		record_tap()
	elif event.is_action_pressed("ui_accept") and offsets.size() >= taps_needed:
		GameState.save_input_offset(result_ms)
		get_tree().change_scene_to_file("res://scenes/Title.tscn")

func record_tap() -> void:
	if conductor.counting_in or offsets.size() >= taps_needed:
		return
	var beat: float = conductor.seconds_to_beat(conductor.song_position)
	var nearest: float = roundf(beat)
	var offset_ms: float = (conductor.song_position - conductor.beat_to_seconds(nearest)) * 1000.0
	offsets.append(offset_ms)
	$Info.text = "TAP %d / %d\nLAST %+d ms" % [offsets.size(), taps_needed, roundi(offset_ms)]
	if offsets.size() >= taps_needed:
		finish()

func finish() -> void:
	var sorted: Array[float] = offsets.duplicate()
	sorted.sort()
	result_ms = sorted[int(sorted.size() / 2.0)]
	$Info.text = "YOUR OFFSET: %+d ms\nENTER: SAVE   ESC: CANCEL" % roundi(result_ms)
	conductor.stop()
