import re
import unittest

"""
問題:
入力として、複数のHTMLタグを含む文字列が与えられます。以下の条件に従って、正規表現を使ってすべての <a> タグの href 属性の値を抽出してください。

条件:
<a> タグの href 属性は、href="URL" の形式で含まれます。
URL はダブルクオートで囲まれており、ダブルクオート内にURLが含まれます。
抽出対象は href 属性の値のみです。
"""
def extract_htmls(text: str) -> bool:
    
    pattern = r'^<a href=\"(?P<link>[\w:/.-]+)\">.*</a>$'          
    res = re.match(pattern, text)

    if res:
        return res.group("link")
    else:
        return "マッチ無し"
    
class TestExtractUrls(unittest.TestCase):
    def test_valid_emails(self):
        html_text = ['<a href="http://example.com">Example</a>', '<a href="https://test.com">Test</a>', '<a href="http://another-example.org">Another Example</a>']
        expected = ["http://example.com", "https://test.com", "http://another-example.org"]

        result = []

        for i in html_text:
            res = extract_htmls(i)
            if res != "マッチ無し":
                result.append(res)

        self.assertEqual(result, expected)

def main():
    tee = TestExtractUrls()
    tee.test_valid_emails()

if __name__ == "__main__":
    # main()
    unittest.main()

