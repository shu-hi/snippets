<?php
/**
*割と万能のsql実行php
*   @param object $c_dbi DB接続インスタンス
*	@param string $sql クエリ文字列のもと
*	@param array $params プリペアードステートメントに渡す値の配列[型、値...]ex)["sis","foo",2,"bar"]
*	@return string 実行結果
*/
function F_exec_sql($c_dbi, $sql, $params) {
    try {
        $c_dbi->DB_on();
        $stmt = $c_dbi->G_DB->prepare($sql);
        if (!$stmt) {
            throw new Exception("Prepare failed: " . $c_dbi->G_DB->error);
        }

        $types = array_shift($params);
        
        // 参照渡し用の配列作成
        $refs = [];
        foreach ($params as $key => $value) {
            $refs[$key] = &$params[$key];
        }

        // bind_paramの引数は型＋参照配列を結合して渡す
        array_unshift($refs, $types);

        if (!call_user_func_array([$stmt, 'bind_param'], $refs)) {
            throw new Exception("bind_param failed: " . $stmt->error);
        }

        
        if (!$stmt->execute()) {
            throw new Exception("execute failed: " . $stmt->error);
        }
        $result = $stmt->get_result();
        if ($result !== false) {
            $rows = $result->fetch_all(MYSQLI_ASSOC);
            return json_encode(['success' => true, 'data' => $rows]);
        } else {
            return json_encode(['success' => true, 'data' => 'SQL executed successfully.']);
        }
    } catch (Exception $e) {
        error_log("Error in execute sql: " . $e->getMessage());
        return json_encode(['success' => false, 'data' => $e->getMessage()]);
    }
}
