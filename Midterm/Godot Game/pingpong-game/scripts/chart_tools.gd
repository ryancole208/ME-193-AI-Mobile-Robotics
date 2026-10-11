class_name ChartTools

static func pattern_for_bar(chart: Dictionary, bar: int) -> Array:
	for section in chart["sections"]:
		if bar >= section["bars"][0] and bar <= section["bars"][1]:
			return section["pattern"]
	return chart["pattern"]
