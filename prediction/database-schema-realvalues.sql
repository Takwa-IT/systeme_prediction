-- ============================================
-- SCRIPT DE MISE À JOUR - VALEURS RÉELLES
-- ============================================
-- Exécuter avec: mysql -u root < database-schema-realvalues.sql

-- 1. Table valeur_reelle_parametre (création complète)
CREATE TABLE IF NOT EXISTS `valeur_reelle_parametre` (
  `id_valeur_reelle_parametre` BIGINT NOT NULL AUTO_INCREMENT,
  `execution_production_id` BIGINT NOT NULL,
  `parametre_etape_id` BIGINT NOT NULL,
  `valeur_reelle` DOUBLE NOT NULL COMMENT 'Valeur réelle mesurée',
  `unite_mesure` VARCHAR(50) COMMENT 'Unité de mesure (°C, min, tr/min, bar, %)',
  `valeur_estimee` DOUBLE COMMENT 'Valeur estimée par le guide',
  `deviation` DOUBLE COMMENT 'Déviation en pourcentage',
  `qualite_deviation` VARCHAR(20) COMMENT 'FAIBLE, MODÉRÉE, IMPORTANTE',
  `date_creation` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `date_modification` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  
  PRIMARY KEY (`id_valeur_reelle_parametre`),
  KEY `idx_execution` (`execution_production_id`),
  KEY `idx_parametre` (`parametre_etape_id`),
  KEY `idx_date` (`date_creation`),
  KEY `idx_qualite_deviation` (`qualite_deviation`),
  
  CONSTRAINT `fk_valeur_execution` FOREIGN KEY (`execution_production_id`) 
    REFERENCES `execution_production` (`id_execution_production`) ON DELETE CASCADE,
  CONSTRAINT `fk_valeur_parametre` FOREIGN KEY (`parametre_etape_id`) 
    REFERENCES `parametre_etape` (`id_parametre_etape`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2. Ajouter colonnes à execution_production si absent
ALTER TABLE `execution_production` ADD COLUMN IF NOT EXISTS `temperature_malaxage_c` DOUBLE;
ALTER TABLE `execution_production` ADD COLUMN IF NOT EXISTS `duree_malaxage_min` DOUBLE;
ALTER TABLE `execution_production` ADD COLUMN IF NOT EXISTS `vitesse_decanteur_tr_min` DOUBLE;
ALTER TABLE `execution_production` ADD COLUMN IF NOT EXISTS `pression_extraction_bar` DOUBLE;

-- 3. Table d'audit pour le feedback loop
CREATE TABLE IF NOT EXISTS `valeur_reelle_audit` (
  `id_audit` BIGINT NOT NULL AUTO_INCREMENT,
  `id_valeur_reelle_parametre` BIGINT NOT NULL,
  `action` VARCHAR(50) COMMENT 'CRÉÉE, MODIFIÉE, SUPPRIMÉE',
  `ancienne_valeur` DOUBLE,
  `nouvelle_valeur` DOUBLE,
  `utilisateur` VARCHAR(255),
  `date_action` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  
  PRIMARY KEY (`id_audit`),
  KEY `idx_valeur` (`id_valeur_reelle_parametre`),
  KEY `idx_date_action` (`date_action`),
  
  CONSTRAINT `fk_audit_valeur` FOREIGN KEY (`id_valeur_reelle_parametre`)
    REFERENCES `valeur_reelle_parametre` (`id_valeur_reelle_parametre`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 4. Vue pour statistiques de feedback
CREATE OR REPLACE VIEW `v_feedback_statistics` AS
SELECT 
  e.`id_execution_production`,
  e.`reference`,
  COUNT(vr.`id_valeur_reelle_parametre`) as nb_parametres,
  AVG(ABS(vr.`deviation`)) as deviation_moyenne,
  MAX(ABS(vr.`deviation`)) as deviation_max,
  SUM(CASE WHEN vr.`qualite_deviation` = 'FAIBLE' THEN 1 ELSE 0 END) as nb_faibles,
  SUM(CASE WHEN vr.`qualite_deviation` = 'MODÉRÉE' THEN 1 ELSE 0 END) as nb_moderees,
  SUM(CASE WHEN vr.`qualite_deviation` = 'IMPORTANTE' THEN 1 ELSE 0 END) as nb_importantes,
  e.`date_creation`,
  e.`statut`
FROM `execution_production` e
LEFT JOIN `valeur_reelle_parametre` vr ON e.`id_execution_production` = vr.`execution_production_id`
GROUP BY e.`id_execution_production`, e.`reference`, e.`date_creation`, e.`statut`;

-- 5. Trigger pour audit
DELIMITER //

DROP TRIGGER IF EXISTS `trg_valeur_reelle_insert` //
CREATE TRIGGER `trg_valeur_reelle_insert` 
AFTER INSERT ON `valeur_reelle_parametre`
FOR EACH ROW
BEGIN
  INSERT INTO `valeur_reelle_audit` (
    `id_valeur_reelle_parametre`, `action`, `nouvelle_valeur`, `utilisateur`
  ) VALUES (
    NEW.`id_valeur_reelle_parametre`, 'CRÉÉE', NEW.`valeur_reelle`, USER()
  );
END//

DROP TRIGGER IF EXISTS `trg_valeur_reelle_update` //
CREATE TRIGGER `trg_valeur_reelle_update`
AFTER UPDATE ON `valeur_reelle_parametre`
FOR EACH ROW
BEGIN
  IF OLD.`valeur_reelle` != NEW.`valeur_reelle` THEN
    INSERT INTO `valeur_reelle_audit` (
      `id_valeur_reelle_parametre`, `action`, `ancienne_valeur`, `nouvelle_valeur`, `utilisateur`
    ) VALUES (
      NEW.`id_valeur_reelle_parametre`, 'MODIFIÉE', OLD.`valeur_reelle`, NEW.`valeur_reelle`, USER()
    );
  END IF;
END//

DELIMITER ;

-- 6. Index supplémentaires pour performance
CREATE INDEX IF NOT EXISTS `idx_execution_date` ON `execution_production` (`id_execution_production`, `date_creation`);
CREATE INDEX IF NOT EXISTS `idx_valeur_creation` ON `valeur_reelle_parametre` (`date_creation`, `qualite_deviation`);

-- ============================================
-- VÉRIFICATION
-- ============================================

SELECT 'Migration complétée' as status;
SELECT COUNT(*) as nb_valeurs_reelles FROM `valeur_reelle_parametre`;
SELECT COUNT(*) as nb_executions FROM `execution_production`;

SHOW TABLES LIKE 'valeur_reelle%';
SHOW VIEWS LIKE 'v_feedback%';

-- Afficher la structure
DESCRIBE `valeur_reelle_parametre`;
DESCRIBE `valeur_reelle_audit`;
