extends CanvasLayer

const FADE_TIME := 0.15
var rect := ColorRect.new()
var busy: bool = false

func _ready() -> void:
	layer = 20
	rect.color = Color("000000")
	rect.modulate.a = 0.0
	rect.mouse_filter = Control.MOUSE_FILTER_IGNORE
	rect.set_anchors_preset(Control.PRESET_FULL_RECT)
	add_child(rect)

func go(path: String) -> void:
	if busy:
		return
	busy = true
	var fade_out := create_tween()
	fade_out.tween_property(rect, "modulate:a", 1.0, FADE_TIME)
	await fade_out.finished
	get_tree().change_scene_to_file(path)
	var fade_in := create_tween()
	fade_in.tween_property(rect, "modulate:a", 0.0, FADE_TIME)
	await fade_in.finished
	busy = false
