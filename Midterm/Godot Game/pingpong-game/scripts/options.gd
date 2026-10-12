extends Control

@onready var music_slider: HSlider = $Center/MusicRow/MusicSlider
@onready var sfx_slider: HSlider = $Center/SfxRow/SfxSlider

func _ready() -> void:
	music_slider.value = GameState.music_volume
	sfx_slider.value = GameState.sfx_volume
	music_slider.value_changed.connect(_on_music_changed)
	sfx_slider.value_changed.connect(_on_sfx_changed)
	music_slider.grab_focus()
	
	
func _on_music_changed(value: float) -> void:
	GameState.music_volume = value
	GameState.apply_volumes()

func _on_sfx_changed(value: float) -> void:
	GameState.sfx_volume = value
	GameState.apply_volumes()
	$TestSound.play()

func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_cancel"):
		GameState.save_volumes()
		get_tree().change_scene_to_file("res://scenes/Title.tscn")
