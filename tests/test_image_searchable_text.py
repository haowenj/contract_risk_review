from image_searchable_text import image_to_searchable_text


def test_bank_image_searchable_text_is_deterministic_and_omits_path():
    image = {
        "img_path": "images/secret.jpg",
        "image_type": "bank_account",
        "structured_data": {
            "account_name": "甲有限公司",
            "account_number": "110914414810101",
            "bank_name": "中国甲银行",
            "bank_branch": None,
        },
        "verification_status": "verified",
    }
    text = image_to_searchable_text(image)
    assert text == (
        "银行账户信息。\n"
        "户名：甲有限公司。\n"
        "开户银行：中国甲银行。\n"
        "银行账号：110914414810101。\n"
        "OCR校验：已核验。"
    )
    assert "secret.jpg" not in text


def test_identity_searchable_text_formats_validity_and_conflict_warning():
    text = image_to_searchable_text(
        {
            "image_type": "identity_card",
            "structured_data": {
                "name": "张三",
                "id_number": "11010119900101123X",
                "valid_from": "2020.01.01",
                "valid_to": "2030.01.01",
            },
            "verification_status": "conflict",
        }
    )
    assert "姓名：张三。" in text
    assert "身份证号码：11010119900101123X。" in text
    assert "有效期限：2020.01.01 至 2030.01.01。" in text
    assert "关键字段存在冲突，需要人工核验" in text


def test_general_image_searchable_text_uses_visible_text_and_description():
    text = image_to_searchable_text(
        {
            "image_type": "general",
            "structured_data": {
                "visible_text": "印章",
                "content_description": "红色圆形印章",
            },
            "verification_status": "not_required",
        }
    )
    assert text == "图片内容：红色圆形印章。\n可见文字：印章。"


def test_empty_general_result_has_no_searchable_text():
    assert (
        image_to_searchable_text(
            {
                "image_type": "general",
                "structured_data": {
                    "visible_text": None,
                    "content_description": "",
                },
                "verification_status": "not_required",
            }
        )
        is None
    )
