extends Node
@export var hit_sounds_enabled: bool = false

func _on_scheduler_opponent_hit() -> void:
	if not hit_sounds_enabled: return
	$HitOpponent.play()

func _on_scheduler_judged(grade: String, _offset_ms: float) -> void:
	if not hit_sounds_enabled: return
	if grade == "Miss":
		$Miss.play()

func _on_scheduler_player_hit() -> void:
	if not hit_sounds_enabled: return
	$HitPlayer.play()

func _on_scheduler_cue() -> void:
	$Cue.play()
