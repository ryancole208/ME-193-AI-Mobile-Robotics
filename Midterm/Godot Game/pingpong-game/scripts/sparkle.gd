extends CPUParticles2D

@export var ball: Node2D
var last_grade: String = ""

func _on_scheduler_judged(grade: String, _offset_ms: float) -> void:
	last_grade = grade

func _on_scheduler_player_hit() -> void:
	if last_grade == "Perfect":
		global_position = ball.get_node("Sprite").global_position
		restart()
