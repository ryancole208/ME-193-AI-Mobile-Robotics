extends Node

const LISTEN_PORT := 4242
const SEND_PORT := 4243
var out := PacketPeerUDP.new()
var udp := PacketPeerUDP.new()
signal state_received(x: float, roll: float, face: String)
signal swing_received(age_s: float, strength: float)
signal status_received(motor: bool, pose: bool)

var last_state_msec: int = -100000
var last_status_msec: int = -100000
var status_label := Label.new()

func is_active() -> bool:
	return Time.get_ticks_msec() - last_state_msec < 500

func _ready() -> void:
	var err: Error = udp.bind(LISTEN_PORT, "127.0.0.1")
	if err != OK:
		push_error("Bridge: could not bind port %d (error %d)" % [LISTEN_PORT, err])
	out.set_dest_address("127.0.0.1", SEND_PORT)
	var layer := CanvasLayer.new()
	layer.layer = 10
	add_child(layer)
	status_label.size = Vector2(380, 10)
	status_label.position = Vector2(0, 2)
	status_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	status_label.add_theme_color_override("font_color", Color("FFF1E8"))
	status_label.add_theme_color_override("font_outline_color", Color("000000"))
	status_label.add_theme_constant_override("outline_size", 1)
	layer.add_child(status_label)
	status_received.connect(_on_status)

func _process(_delta: float) -> void:
	while udp.get_available_packet_count() > 0:
		var text: String = udp.get_packet().get_string_from_utf8()
		var msg: Variant = JSON.parse_string(text)
		if msg is Dictionary:
			handle(msg)
	if Time.get_ticks_msec() - last_status_msec > 2000:
		status_label.text = ""
func handle(msg: Dictionary) -> void:
	match msg.get("type", ""):
		"state":
			last_state_msec = Time.get_ticks_msec()
			state_received.emit(float(msg["x"]), float(msg["roll"]), str(msg["face"]))
		"swing":
			swing_received.emit(float(msg.get("age_ms", 0.0)) / 1000.0, float(msg.get("strength", 0.0)))
		"status":
			status_received.emit(bool(msg["motor"]), bool(msg["pose"]))

func send(msg: Dictionary) -> void:
	out.put_packet(JSON.stringify(msg).to_utf8_buffer())

func _on_status(motor: bool, pose: bool) -> void:
	last_status_msec = Time.get_ticks_msec()
	status_label.text = "PADDLE %s  CAM %s" % ["OK" if motor else "--", "OK" if pose else "--"]
