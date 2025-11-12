import { useState } from 'react'
import { Button, Drawer, Space, Table, Tag, Typography } from 'antd'
import type { TableColumnsType } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { PageContainer } from '@/components/layout/PageContainer'
import { getMockDatasets } from '@/services/mockApi'
import type { DatasetItem } from '@/services/mockApi'

export const DatasetsPage = () => {
  const { data: datasets } = useQuery({ queryKey: ['datasets'], queryFn: getMockDatasets })
  const [selected, setSelected] = useState<DatasetItem | null>(null)

  const columns: TableColumnsType<DatasetItem> = [
    { title: '名称', dataIndex: 'name' },
    { title: '规模', dataIndex: 'size', width: 120 },
    {
      title: '面向阶段',
      dataIndex: 'stage',
      width: 160,
      render: (stage: DatasetItem['stage']) => <Tag>{stage}</Tag>,
    },
    { title: '领域', dataIndex: 'domain', width: 160 },
    {
      title: '操作',
      dataIndex: 'id',
      width: 200,
      render: (_, record) => (
        <Space>
          <Button size="small" onClick={() => setSelected(record)}>
            预览
          </Button>
          <Button size="small" type="text">
            指派
          </Button>
        </Space>
      ),
    },
  ]

  return (
    <PageContainer
      title="数据集中心"
      description="集中托管 LLaMA-Factory data config，同步预览与权限控制。"
      extra={<Button type="primary">上传数据集</Button>}
    >
      <Table rowKey="id" columns={columns} dataSource={datasets} pagination={false} />
      <Drawer width={480} destroyOnClose open={!!selected} onClose={() => setSelected(null)} title={selected?.name}>
        {selected?.samples.map((item, index) => (
          <div key={index} className="dataset-sample">
            {Object.entries(item).map(([key, value]) => (
              <Typography.Paragraph key={key}>
                <Typography.Text strong>{key}</Typography.Text>: {value}
              </Typography.Paragraph>
            ))}
          </div>
        ))}
      </Drawer>
    </PageContainer>
  )
}
