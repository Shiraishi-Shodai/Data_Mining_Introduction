<?php

    $test_str ="<html><h1>ここ大見出し</h1><p>ここに文章</p></html>";
    $pattern = '/(<h.*?>)<p>(.*)<\/p>/';

    if(preg_match_all($pattern, $test_str,$m)) {
        echo "マッチします。\n";
        print_r($m);
    }else{
        echo "マッチしません。";
    }
?>