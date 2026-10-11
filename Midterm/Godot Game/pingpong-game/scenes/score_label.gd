extends Label

const GRADE_POINTS: Dictionary = {"Perfect": 1.0, "Good": 0.5, "Miss": 0.0}
var points: float = 0.0
var judged_count: int = 0
var streak: int = 0
var best_streak: int = 0
var last_grade: String = ""

func _ready() -> void:
	update_text()

func update_text() -> void:
	var accuracy: float = 100.0 * points / judged_count if judged_count > 0 else 100.0
	text = "%s\nAcc %.1f%%\nStreak %d" % [last_grade, accuracy, streak]


func _on_scheduler_judged(grade: String, _offset_ms: float) -> void:
	last_grade = grade
	points += GRADE_POINTS[grade]
	judged_count += 1
	streak = 0 if grade == "Miss" else streak + 1
	best_streak = maxi(best_streak, streak)
	update_text()
