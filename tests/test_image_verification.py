from image_verification import verify_image_data


def test_bank_account_number_exact_match_after_separator_normalization():
    result = verify_image_data(
        "bank_account",
        {
            "account_name": "甲有限公司",
            "account_number": "1109 1441-4810 101",
            "bank_name": "中国甲银行",
            "bank_branch": None,
        },
        "户名：甲有限公司\n开户银行：中国甲银行\n账号：110914414810101",
    )
    assert result.status == "verified"
    assert result.details["account_number"]["status"] == "verified"


def test_one_digit_account_difference_is_conflict():
    result = verify_image_data(
        "bank_account",
        {
            "account_name": "甲有限公司",
            "account_number": "110914414810101",
            "bank_name": "中国甲银行",
            "bank_branch": None,
        },
        "户名：甲有限公司\n开户银行：中国甲银行\n账号：110914414810107",
    )
    assert result.status == "conflict"
    assert result.details["account_number"]["status"] == "conflict"


def test_identity_without_ocr_candidate_is_insufficient():
    result = verify_image_data(
        "identity_card",
        {
            "name": "张三",
            "id_number": "11010119900101123X",
            "valid_from": None,
            "valid_to": None,
        },
        "模糊文字",
    )
    assert result.status == "insufficient"


def test_identity_x_case_and_reliable_dates_are_verified():
    result = verify_image_data(
        "identity_card",
        {
            "name": "张三",
            "id_number": "11010119900101123x",
            "valid_from": "2020.01.01",
            "valid_to": "2030.01.01",
        },
        "姓名：张三\n公民身份号码：11010119900101123X\n"
        "有效期限：2020年01月01日-2030年01月01日",
    )
    assert result.status == "verified"
    assert result.details["id_number"]["status"] == "verified"
    assert result.details["valid_from"]["status"] == "verified"


def test_general_does_not_use_ocr():
    result = verify_image_data(
        "general",
        {"visible_text": "印章", "content_description": "红色印章"},
        "完全不同的 OCR",
    )
    assert result.status == "not_required"
    assert result.details == {}
