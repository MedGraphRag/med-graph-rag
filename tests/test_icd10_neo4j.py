import pytest

# ============================================================
# Helpers
# ============================================================

def get_codes(driver):
    query = """
    MATCH (n:ICD10)
    RETURN collect(n.code) AS codes
    """

    with driver.session() as session:
        record = session.run(query).single()

    return record["codes"]


def get_node(driver, code: str):
    query = """
    MATCH (n:ICD10 {code: $code})
    RETURN n
    """

    with driver.session() as session:
        record = session.run(
            query,
            code=code,
        ).single()

    return record["n"] if record else None


# ============================================================
# 1. Connection
# ============================================================

def test_neo4j_connection(neo4j_driver):
    """
    Neo4j должен быть доступен.
    """

    neo4j_driver.verify_connectivity()


# ============================================================
# 2. ICD10 nodes exist
# ============================================================

def test_icd10_nodes_exist(neo4j_driver):
    """
    После импорта в БД должны существовать ICD10 nodes.
    """

    query = """
    MATCH (n:ICD10)
    RETURN count(n) AS count
    """

    with neo4j_driver.session() as session:
        record = session.run(query).single()

    assert record["count"] > 0


# ============================================================
# 3. Known codes exist
# ============================================================

@pytest.mark.parametrize(
    "code",
    [
        "I",
        "I10",
        "I11",
        "I11.0",
    ],
)
def test_known_icd10_codes_exist(neo4j_driver, code):
    """
    Проверяем наличие конкретных узлов.
    """

    node = get_node(
        neo4j_driver,
        code,
    )

    assert node is not None
    assert node["code"] == code


# ============================================================
# 4. Required node properties
# ============================================================

def test_icd10_node_has_required_properties(neo4j_driver):
    """
    Каждый ICD10 node должен иметь:
      - code
      - name
      - level
    """

    query = """
    MATCH (n:ICD10)
    WHERE n.code IS NULL
       OR n.name IS NULL
       OR n.level IS NULL
    RETURN count(n) AS invalid
    """

    with neo4j_driver.session() as session:
        record = session.run(query).single()

    assert record["invalid"] == 0


# ============================================================
# 5. Codes are unique
# ============================================================

def test_icd10_codes_are_unique(neo4j_driver):
    """
    Один code должен соответствовать максимум одному узлу.
    """

    query = """
    MATCH (n:ICD10)
    WITH n.code AS code, count(n) AS count
    WHERE count > 1
    RETURN count(*) AS duplicates
    """

    with neo4j_driver.session() as session:
        record = session.run(query).single()

    assert record["duplicates"] == 0


# ============================================================
# 6. PARENT_OF relationships exist
# ============================================================

def test_parent_relationships_exist(neo4j_driver):
    """
    В графе должны существовать иерархические связи.
    """

    query = """
    MATCH ()-[r:PARENT_OF]->()
    RETURN count(r) AS count
    """

    with neo4j_driver.session() as session:
        record = session.run(query).single()

    assert record["count"] > 0


# ============================================================
# 7. Known hierarchy
# ============================================================

def test_i11_parent_is_i10_or_expected_parent(neo4j_driver):
    """
    Проверяем конкретную связь в иерархии.

    Важно:
    фактический parent берётся из импортированного JSON.
    Если для I11 в твоём dataset другой непосредственный
    parent, этот тест нужно изменить.
    """

    query = """
    MATCH (parent:ICD10)-[:PARENT_OF]->(child:ICD10)
    WHERE child.code = "I11"
    RETURN parent.code AS parent_code
    """

    with neo4j_driver.session() as session:
        record = session.run(query).single()

    assert record is not None
    assert record["parent_code"] == "I10-I15"


# ============================================================
# 8. I11.0 has a parent
# ============================================================

def test_i11_0_has_parent(neo4j_driver):
    """
    I11.0 должен иметь непосредственного родителя.
    """

    query = """
    MATCH (parent:ICD10)-[:PARENT_OF]->(child:ICD10)
    WHERE child.code = "I11.0"
    RETURN parent.code AS parent_code
    """

    with neo4j_driver.session() as session:
        record = session.run(query).single()

    assert record is not None
    assert record["parent_code"] == "I11"


# ============================================================
# 9. Transitive parent search
# ============================================================

def test_transitive_parent_search(neo4j_driver):
    """
    Главный тест из задания.

    Для I11.0 должны находиться все предки
    через произвольную глубину PARENT_OF.
    """

    query = """
    MATCH (child:ICD10 {code: "I11.0"})
          <-[:PARENT_OF*1..]-
          (parent:ICD10)

    RETURN collect(parent.code) AS parents
    """

    with neo4j_driver.session() as session:
        record = session.run(query).single()

    parents = set(record["parents"])

    assert "I11" in parents
    assert "I10-I15" in parents
    assert "IX" in parents


# ============================================================
# 10. Transitive chain reaches class
# ============================================================

def test_transitive_search_reaches_class(neo4j_driver):
    """
    Проверяем, что поиск по цепочке действительно
    доходит до класса болезней.
    """

    query = """
    MATCH path = (
        child:ICD10 {code: "I11.0"}
    )-[:PARENT_OF*1..]->(
        parent:ICD10
    )

    WHERE parent.level = "class"

    RETURN parent.code AS class_code
    LIMIT 1
    """

    with neo4j_driver.session() as session:
        record = session.run(query).single()

    assert record is not None
    assert record["class_code"] == "IX"


# ============================================================
# 11. Every non-root node has a parent
# ============================================================

def test_every_non_root_node_has_parent(neo4j_driver):
    """
    Каждый узел, кроме корневого, должен иметь
    входящую PARENT_OF связь.
    """

    query = """
    MATCH (n:ICD10)
    WHERE n.parent_code IS NOT NULL
      AND NOT (()-[:PARENT_OF]->(n))
    RETURN count(n) AS invalid
    """

    with neo4j_driver.session() as session:
        record = session.run(query).single()

    assert record["invalid"] == 0


# ============================================================
# 12. Root nodes have no parent
# ============================================================

def test_root_nodes_have_no_parent(neo4j_driver):
    """
    Узлы без parent_code не должны иметь
    входящей PARENT_OF связи.
    """

    query = """
    MATCH (n:ICD10)
    WHERE n.parent_code IS NULL
      AND (()-[:PARENT_OF]->(n))
    RETURN count(n) AS invalid
    """

    with neo4j_driver.session() as session:
        record = session.run(query).single()

    assert record["invalid"] == 0


# ============================================================
# 13. parent_code matches graph relationship
# ============================================================

def test_parent_code_matches_relationship(neo4j_driver):
    """
    parent_code в properties должен совпадать
    с фактическим родителем в графе.
    """

    query = """
    MATCH (child:ICD10)
    WHERE child.parent_code IS NOT NULL

    OPTIONAL MATCH (parent:ICD10)-[:PARENT_OF]->(child)

    WHERE parent IS NULL
       OR parent.code <> child.parent_code

    RETURN count(child) AS invalid
    """

    with neo4j_driver.session() as session:
        record = session.run(query).single()

    assert record["invalid"] == 0


# ============================================================
# 14. No orphan relationships
# ============================================================

def test_no_orphan_relationships(neo4j_driver):
    """
    Все PARENT_OF relationships должны соединять
    существующие ICD10 nodes.

    В Neo4j это гарантируется самим графом,
    но тест документирует требование.
    """

    query = """
    MATCH (parent)-[r:PARENT_OF]->(child)
    WHERE NOT parent:ICD10
       OR NOT child:ICD10
    RETURN count(r) AS invalid
    """

    with neo4j_driver.session() as session:
        record = session.run(query).single()

    assert record["invalid"] == 0


# ============================================================
# 15. Valid ICD10 levels
# ============================================================

def test_icd10_levels_are_valid(neo4j_driver):
    """
    level должен быть одним из допустимых значений.
    """

    query = """
    MATCH (n:ICD10)
    WHERE n.level NOT IN [
        "class",
        "block",
        "category",
        "subcategory"
    ]
    RETURN count(n) AS invalid
    """

    with neo4j_driver.session() as session:
        record = session.run(query).single()

    assert record["invalid"] == 0


# ============================================================
# 16. Class nodes are roots
# ============================================================

def test_class_nodes_are_roots(neo4j_driver):
    """
    class-узлы не должны иметь родителей.
    """

    query = """
    MATCH (n:ICD10)
    WHERE n.level = "class"
      AND (()-[:PARENT_OF]->(n))
    RETURN count(n) AS invalid
    """

    with neo4j_driver.session() as session:
        record = session.run(query).single()

    assert record["invalid"] == 0


# ============================================================
# 17. No self relationships
# ============================================================

def test_no_self_parent_relationships(neo4j_driver):
    """
    Узел не может быть родителем самого себя.
    """

    query = """
    MATCH (n:ICD10)-[:PARENT_OF]->(n)
    RETURN count(n) AS invalid
    """

    with neo4j_driver.session() as session:
        record = session.run(query).single()

    assert record["invalid"] == 0


# ============================================================
# 18. No cycles
# ============================================================

def test_no_hierarchy_cycles(neo4j_driver):
    """
    В иерархии МКБ-10 не должно быть циклов.
    """

    query = """
    MATCH path = (n:ICD10)-[:PARENT_OF*1..]->(n)
    RETURN count(path) AS cycles
    """

    with neo4j_driver.session() as session:
        record = session.run(query).single()

    assert record["cycles"] == 0


# ============================================================
# 19. Every parent exists
# ============================================================

def test_every_parent_exists(neo4j_driver):
    """
    Если parent_code указан, соответствующий
    ICD10 node обязан существовать.
    """

    query = """
    MATCH (child:ICD10)
    WHERE child.parent_code IS NOT NULL

    OPTIONAL MATCH (parent:ICD10 {
        code: child.parent_code
    })

    WITH child, parent
    WHERE parent IS NULL

    RETURN count(child) AS invalid
    """

    with neo4j_driver.session() as session:
        record = session.run(query).single()

    assert record["invalid"] == 0


# ============================================================
# 20. Complete hierarchy for I11.0
# ============================================================

def test_i11_0_complete_hierarchy(neo4j_driver):
    """
    Проверяем конкретную цепочку:

        IX
        ↓
        I10-I15
        ↓
        I11
        ↓
        I11.0
    """

    query = """
    MATCH path = (
        root:ICD10 {code: "I"}
    )-[:PARENT_OF*1..]->(
        leaf:ICD10 {code: "I11.0"}
    )

    RETURN path
    """

    with neo4j_driver.session() as session:
        record = session.run(query).single()

    assert record is not None