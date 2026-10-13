extends Label

@export var show_time: float = 0.65
var time_left: float = 0.0
var tween: Tween
const COLORS: Dictionary = {
	"Perfect": Color("FFEC27"), "Good": Color("FFA300"), "Miss": Color("FF004D")
}

func _ready() -> void:
	visible = false

func _process(delta: float) -> void:
	if time_left > 0.0:
		time_left -= delta
		if time_left <= 0.0:
			visible = false

func _on_scheduler_judged(grade: String, _offset_ms: float) -> void:
	text = grade
	modulate = COLORS[grade]
	time_left = show_time
	visible = true
	pivot_offset = size / 2.0
	scale = Vector2(1.6, 1.6)
	if tween:
		tween.kill()
	tween = create_tween()
	tween.tween_property(self, "scale", Vector2.ONE, 0.15) \
		.set_trans(Tween.TRANS_BACK).set_ease(Tween.EASE_OUT)
