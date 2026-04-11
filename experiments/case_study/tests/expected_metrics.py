EMAIL_EXPECTED_ERROR_COUNTS = {
    "baseline-email.json": {
        (True, True, True): 311,
        (True, True, False): 311,
        (True, False, True): 311,
        (True, False, False): 311,
        (False, True, True): 311,
        (False, True, False): 311,
        (False, False, True): 311,
        (False, False, False): 311,
    },
    "diverse-email.json": {
        (True, True, True): 314,
        (True, True, False): 315,
        (True, False, True): 327,
        (True, False, False): 327,
        (False, True, True): 321,
        (False, True, False): 322,
        (False, False, True): 333,
        (False, False, False): 333,
    },
}


WEBCOLORS_EXPECTED_ERROR_COUNTS = {
    "baseline-css-color.json": {
        "html5_parse_legacy_color": 4,
        "html5_parse_simple_color": 1000,
        "hex_to_name": 1000,
        "hex_to_rgb": 1000,
        "hex_to_rgb_percent": 1000,
        "name_to_hex": 788,
        "name_to_rgb": 788,
        "name_to_rgb_percent": 788,
    },
    "diverse-css-color.json": {
        "html5_parse_legacy_color": 13,
        "html5_parse_simple_color": 998,
        "hex_to_name": 990,
        "hex_to_rgb": 963,
        "hex_to_rgb_percent": 963,
        "name_to_hex": 392,
        "name_to_rgb": 392,
        "name_to_rgb_percent": 392,
    },
}
