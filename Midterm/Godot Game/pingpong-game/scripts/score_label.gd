extends Label

const GRADE_POINTS: Dictionary = {"Perfect": 1.0, "Good": 0.5, "Miss": 0.0}
var points: float = 0.0
var judged_count: int = 0
var streak: int = 0
var best_streak: int = 0
var last_grade: String = ""
@export var conductor: Conductor
var counts: Dictionary = {"Perfect": 0, "Good": 0, "Miss": 0}

func _ready() -> void:
	update_text()
	conductor.finished.connect(_on_song_finished)

func update_text() -> void:
	text = "%s\nAcc %.1f%%\nStreak %d" % [last_grade, accuracy(), streak]


func _on_scheduler_judged(grade: String, _offset_ms: float) -> void:
	last_grade = grade
	points += GRADE_POINTS[grade]
	judged_count += 1
	counts[grade] += 1
	streak = 0 if grade == "Miss" else streak + 1
	best_streak = maxi(best_streak, streak)
	update_text()

func accuracy() -> float:
	return 100.0 * points / judged_count if judged_count > 0 else 100.0

func _on_song_finished() -> void:
	GameState.results = {
		"title": conductor.chart["title"],
		"accuracy": accuracy(),
		"counts": counts,
		"best_streak": best_streak,
		"full_run": conductor.debug_start_bar == 0,
	}
	get_tree().change_scene_to_file("res://scenes/Results.tscn")
