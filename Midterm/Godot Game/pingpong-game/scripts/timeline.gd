extends Control

@export var range_ms: float = 100.0
var offsets: Array[float] = []

func add_offset(ms: float) -> void:
	offsets.append(ms)
	queue_redraw()

func _draw() -> void:
	var mid_y: float = floorf(size.y / 2.0)
	var center_x: float = floorf(size.x / 2.0)
	draw_rect(Rect2(0, mid_y, size.x, 1), Color("C2C3C7"))
	draw_rect(Rect2(center_x, mid_y - 3, 1, 7), Color("FFF1E8"))
	draw_rect(Rect2(0, mid_y - 2, 1, 5), Color("C2C3C7"))
	draw_rect(Rect2(size.x - 1, mid_y - 2, 1, 5), Color("C2C3C7"))
	for i in offsets.size():
		var t: float = clampf(offsets[i] / range_ms, -1.0, 1.0)
		var x: float = roundf(center_x + t * (center_x - 1.0))
		var color := Color("FFEC27") if i == offsets.size() - 1 else Color("FFA300")
		draw_rect(Rect2(x - 1, mid_y - 1, 3, 3), color)
