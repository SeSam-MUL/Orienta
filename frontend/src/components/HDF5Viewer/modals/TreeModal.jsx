import { useTranslation } from 'react-i18next';
import Modal from './Modal';
import TreeTab from '../tabs/TreeTab';

export default function TreeModal({ isFileOpen, onClose }) {
  const { t } = useTranslation('hdf5viewer');
  return (
    <Modal title={t('treeModal.title')} onClose={onClose}>
      <div style={{ height: 500, display: 'flex' }}>
        <TreeTab isFileOpen={isFileOpen} />
      </div>
    </Modal>
  );
}
