import BatchWorkflow from './BatchWorkflow';

export default function BatchPage({ isActive }) {
  return (
    <div style={{
      display: isActive ? 'flex' : 'none',
      flexDirection: 'column',
      height: '100%',
    }}>
      <BatchWorkflow isActive={isActive} />
    </div>
  );
}
